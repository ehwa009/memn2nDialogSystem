from sklearn import metrics

import data.data_utils as data_utils
import models.memn2n as memn2n
import numpy as np

import argparse
import sys

import pickle as pkl
import tensorflow as tf


DATA_DIR = 'data/dialog-bAbI-tasks/'
P_DATA_DIR = 'data/processed/'
BATCH_SIZE = 16
CKPT_DIR= 'ckpt/'

class InteractiveSession():

    def __init__(self, model, idx2candid, w2idx, n_cand, memory_size):
        self.context = []
        self.u = None
        self.r = None
        self.nid = 1
        self.model = model
        self.idx2candid = idx2candid
        self.w2idx = w2idx
        self.n_cand = n_cand
        self.memory_size = memory_size
        self.model = model

    def reply(self, msg):
        pass

def parse_args(args):
    parser = argparse.ArgumentParser(
                description='Train Model for Goal Oriented Dialog Task : bAbI(6)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-i', '--infer', action='store_true',
                        help='perform inference in an interactive session')
    group.add_argument('--ui', action='store_true',
                        help='interact through web app(flask); do not call this from cmd line')
    group.add_argument('-t', '--train', action='store_true',
                        help='train model')
    group.add_argument('-d', '--prep_data', action='store_true',
                        help='prepare data')
    parser.add_argument('--task_id', required=False, type=int, default=1,
                        help='Task Id in bAbI (6) tasks {1-6}')
    parser.add_argument('--batch_size', required=False, type=int, default=16,
                        help='you know what batch size means!')
    parser.add_argument('--epochs', required=False, type=int, default=200,
                        help='num iteration of training over train set')
    parser.add_argument('--eval_interval', required=False, type=int, default=5,
                        help='num iteration of training over train set')
    parser.add_argument('--log_file', required=False, type=str, default='log.txt',
                        help='enter the name of the log file')
    
    args = vars(parser.parse_args(args))
    return args

def prepare_data(args, task_id):
    # get condidate response (restaurants domain)
    candidates, candid2idx, idx2candid = data_utils.load_candidates(task_id=task_id,
                                                        candidates_f=DATA_DIR + 'dialog-babi-candidates.txt')

    # get train, test, val data
    train, test, val = data_utils.load_dialog_task(
            data_dir = DATA_DIR,
            task_id = task_id,
            candid_dic = candid2idx,
            isOOV = False)

    # get metadata
    metadata = data_utils.build_vocab(train + test + val, candidates)

    # write data to file (pickle을 사용함 이거 빠름)
    data_ = {
            'candidates': candidates,
            'train': train,
            'test': test,
            'val': val
            }  
    with open(P_DATA_DIR + str(task_id) + '.data.pkl', 'wb') as f:
        pkl.dump(data_, f)

    # 메타데이터에 추가 후 저장
    metadata['candid2idx'] = candid2idx
    metadata['idx2candid'] = idx2candid

    with open(P_DATA_DIR + str(task_id) + '.metadata.pkl', 'wb') as f:
        pkl.dump(metadata, f)

def main(args):
    # parse args
    args = parse_args(args)

    # prepare data
    if args['prep_data']:
        print('\n ====== preparing data ====== \n')
        for i in range(1, 7):
            print(' TASK #{}\n'.format(i))
            prepare_data(args, task_id=i)
        sys.exit()
    
    ##################################################################
    # 데이터 준비가 아니면 read data and metadata from pickled files 
    ##################################################################
    with open(P_DATA_DIR + str(args['task_id']) + '.metadata.pkl', 'rb') as f:
        metadata = pkl.load(f)
    with open(P_DATA_DIR + str(args['task_id']) + '.data.pkl', 'rb') as f:
        data_ = pkl.load(f)

    # read content of data and metadata
    candidates = data_['candidates']
    candid2idx, idx2candid = metadata['candid2idx'], metadata['idx2candid']

    # read train, test, val data
    train, test, val = data_['train'], data_['test'], data_['val']

    # get more required information from metadata
    sentence_size = metadata['sentence_size']
    w2idx = metadata['w2idx']
    idx2w = metadata['idx2w']
    memory_size = metadata['memory_size']
    vocab_size = metadata['vocab_size']
    n_cand = metadata['n_cand']
    candidate_sentence_size = metadata['candidate_sentence_size']

    # 후보 response들의 백터화
    candidates_vec = data_utils.vectorize_candidates(candidates, w2idx, candidate_sentence_size)

    # create model - memn2n
    model = memn2n.MemN2NDialog(
                batch_size = BATCH_SIZE,
                vocab_size = vocab_size,
                candidates_size = n_cand,
                sentence_size = sentence_size,
                embedding_size = 20,
                candidates_vec = candidates_vec,
                hops = 3
            )

    train, val, test, batches = data_utils.get_batches(train, val, test, metadata, batch_size=BATCH_SIZE)
    
    # training은 여기서 실행 된다.
    if args['train']:
        epochs = args['epochs']
        eval_interval = args['eval_interval']

        print('\n>>> Training started...\n')

        # write log to file
        log_handle = open('log/' + args['log_file'], 'w')
        cost_total = 0.

        for i in range(epochs+1):
            for start, end in batches:
                s = train['s'][start:end]
                q = train['q'][start:end]
                a = train['a'][start:end]
                cost_total += model.batch_fit(s, q, a)

            if i%eval_interval == 0 and i:
                train_preds = batch_predict(model, train['s'], train['q'], len(train['s']), batch_size=BATCH_SIZE)
                val_preds = batch_predict(model, val['s'], val['q'], len(val['s']), batch_size=BATCH_SIZE)
                train_acc = metrics.accuracy_score(np.array(train_preds), train['a'])
                val_acc = metrics.accuracy_score(val_preds, val['a'])
                print('Epoch[{}] : <Accuracy>\n\ttraining : {} \n\tvalidation : {}'.
                                format(i, train_acc, val_acc))
                log_handle.write('{} {} {} {}\n'.format(i, train_acc, val_acc, 
                    cost_total/(eval_interval*len(batches))))
                cost_total =0.

                model.saver.save(model._sess, CKPT_DIR + '{}/memn2n_model.ckpt'.format(args['task_id']), 
                        global_step=i)
        
        log_handle.close()
                

'''

    run prediction on dataset

'''
def batch_predict(model, S,Q,n, batch_size):
    preds = []
    for start in range(0, n, batch_size):
        end = start + batch_size
        s = S[start:end]
        q = Q[start:end]
        pred = model.predict(s, q)
        preds += list(pred)
    return preds




if __name__ == '__main__':
    # main(sys.args[1:])
    main(['--train', '--task_id=1'])