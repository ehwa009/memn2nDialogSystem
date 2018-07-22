import tensorflow as tf

class MemN2NDialog(object):

    def __init__(self, batch_size, vocab_size, candidates_size, sentence_size, embedding_size,
            candidates_vec,
            hops=3,
            max_grad_norm=80.0,
            nonlin=None,
            initializer=tf.random_normal_initializer(stddev=0.1),
            optimizer=tf.train.AdamOptimizer(learning_rate=1e-3, epsilon=1e-8),
            session=tf.Session(),
            name='MemN2N'):

        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._candidates_size = candidates_size
        self._sentence_size = sentence_size
        self._embedding_size = embedding_size
        self._hops = hops
        self._max_grad_norm = max_grad_norm
        self._nonlin = nonlin
        self._init = initializer
        self._opt = optimizer
        self._name = name
        self._candidates=candidates_vec
        self.saver = None

        self._build_inputs()
        self._build_vars()

        # cross entropy
        logits = self._inference(self._stories, self._queries)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self._answers, name="cross_entropy")
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy_mean")

        loss_op = cross_entropy_mean
    
    def _build_inputs(self):
        self._stories = tf.placeholder(tf.int32, [None, None, self._sentence_size], name="stories")
        self._queries = tf.placeholder(tf.int32, [None, self._sentence_size], name="queries")
        self._answers = tf.placeholder(tf.int32, [None], name="answer")

    def _build_vars(self):
        with tf.variable_scope(self._name):
            nil_word_slot = tf.zeros([1, self._embedding_size])
            A = tf.concat([ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ], 0)
            self.A = tf.Variable(A, name="A")
            self.H = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="H")
            W = tf.concat([ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ], 0)
            self.W = tf.Variable(W, name="W")
        self._nil_vars = set([self.A.name, self.W.name])

    def _inference(self, stories, queries):
        with tf.variable_scope(self._name):
            q_emb = tf.nn.embedding_lookup(self.A, queries)
            u_0 = tf.reduce_sum(q_emb, 1)
            u = [u_0]
            for _ in range(self._hops):
                m_emb = tf.nn.embedding_lookup(self.A, stories)
                m = tf.reduce_sum(m_emb, 2)
                
                u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])
                dotted = tf.reduce_sum(m * u_temp, 2)

                probs = tf.nn.softmax(dotted)
                
                probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])
                c_temp = tf.transpose(m, [0, 2, 1])
                o_k = tf.reduce_sum(c_temp * probs_temp, 2)

                u_k = tf.matmul(u[-1], self.H) + o_k
                # u_k=u[-1]+tf.matmul(o_k,self.H)
                # nonlinearity
                if self._nonlin:
                    u_k = self._nonlin(u_k)

                u.append(u_k)
            candidates_emb=tf.nn.embedding_lookup(self.W, self._candidates)
            candidates_emb_sum=tf.reduce_sum(candidates_emb,1)
            return tf.matmul(u_k,tf.transpose(candidates_emb_sum))