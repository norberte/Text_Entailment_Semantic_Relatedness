import tensorflow as tf
import numpy as np
from scipy.stats.stats import pearsonr

glove_vectors_file = "..\emb_models\glove.6B.300d.txt"

dev_set_file = "..\data\SICK_trial.txt"
train_set_file = "..\data\SICK_train.txt"
test_set_file = "..\data\SICK_test_annotated.txt"

output_dir = "..\output\\"

glove_wordmap = {}
with open(glove_vectors_file, "r", encoding='utf-8') as glove:
    for line in glove:
        name, vector = tuple(line.split(" ", 1))
        glove_wordmap[name] = np.fromstring(vector, sep=" ")

def sentence2sequence(sentence):
    """
    - Turns an input sentence into an (n,d) matrix,
        where n is the number of tokens in the sentence
        and d is the number of dimensions each word vector has.

      Tensorflow doesn't need to be used here, as simply
      turning the sentence into a sequence based off our
      mapping does not need the computational power that
      Tensorflow provides. Normal Python suffices for this task.
    """
    tokens = sentence.lower().split(" ")
    rows = []
    words = []
    # Greedy search for tokens
    for token in tokens:
        i = len(token)
        while len(token) > 0 and i > 0:
            word = token[:i]
            if word in glove_wordmap:
                rows.append(glove_wordmap[word])
                words.append(word)
                token = token[i:]
                i = len(token)
            else:
                i = i - 1
    return rows, words

rnn_size = 128
rnn = tf.nn.rnn_cell.LSTMCell(rnn_size)

# Constants setup
max_hypothesis_length, max_evidence_length = 30, 30
batch_size, vector_size, hidden_size = 256, 300, 128

lstm_size = hidden_size

weight_decay = 0.0001

learning_rate = 0.001

input_p, output_p = 0.5, 0.5

training_iterations_count = 100000

display_step = 10


def fit_to_size(matrix, shape):
    res = np.zeros(shape)
    slices = [slice(0, min(dim, shape[e])) for e, dim in enumerate(matrix.shape)]
    res[tuple(slices)] = matrix[tuple(slices)]
    return res

def split_data_into_scores(file):
    import csv
    with open(file, "r") as data:
        train = csv.DictReader(data, delimiter='\t')
        ids = []
        evi_sentences = []
        hyp_sentences = []
        scores = []
        for row in train:
            ids.append(row["pair_ID"])
            hyp_sentences.append(np.vstack(
                sentence2sequence(row["sentence_A"].lower())[0]))
            evi_sentences.append(np.vstack(
                sentence2sequence(row["sentence_B"].lower())[0]))
            scores.append(float(row["relatedness_score"]))

        hyp_sentences = np.stack([fit_to_size(x, (max_hypothesis_length, vector_size))
                                  for x in hyp_sentences])
        evi_sentences = np.stack([fit_to_size(x, (max_evidence_length, vector_size))
                                  for x in evi_sentences])

        return ids, (hyp_sentences, evi_sentences), np.array(scores).reshape((len(scores), 1))


train_IDs, data_feature_list, correct_scores = split_data_into_scores(train_set_file)

l_h, l_e = max_hypothesis_length, max_evidence_length
N, D, H = batch_size, vector_size, hidden_size
l_seq = l_h + l_e


tf.reset_default_graph()

lstm = tf.nn.rnn_cell.LSTMCell(lstm_size)

lstm_drop =  tf.contrib.rnn.DropoutWrapper(lstm, input_p, output_p)

# N: The number of elements in each of our batches,
#   which we use to train subsets of data for efficiency's sake.
# l_h: The maximum length of a hypothesis, or the second sentence.  This is
#   used because training an RNN is extraordinarily difficult without
#   rolling it out to a fixed length.
# l_e: The maximum length of evidence, the first sentence.  This is used
#   because training an RNN is extraordinarily difficult without
#   rolling it out to a fixed length.
# D: The size of our used GloVe or other vectors.
hyp = tf.placeholder(tf.float32, [None, l_h, D], 'hypothesis')
evi = tf.placeholder(tf.float32, [None, l_e, D], 'evidence')
y = tf.placeholder(tf.float32, [None, 1], 'label')
# hyp: Where the hypotheses will be stored during training.
# evi: Where the evidences will be stored during training.
# y: Where correct scores will be stored during training.

# lstm_size: the size of the gates in the LSTM,
#    as in the first LSTM layer's initialization.
lstm_back = tf.nn.rnn_cell.LSTMCell(lstm_size)
# lstm_back:  The LSTM used for looking backwards
#   through the sentences, similar to lstm.

# input_p: the probability that inputs to the LSTM will be retained at each
#   iteration of dropout.
# output_p: the probability that outputs from the LSTM will be retained at
#   each iteration of dropout.
lstm_drop_back = tf.contrib.rnn.DropoutWrapper(lstm_back, input_p, output_p)
# lstm_drop_back:  A dropout wrapper for lstm_back, like lstm_drop.


fc_initializer = tf.random_normal_initializer(stddev=0.1)
# fc_initializer: initial values for the fully connected layer's weights.
# hidden_size: the size of the outputs from each lstm layer.
#   Multiplied by 2 to account for the two LSTMs.
fc_weight = tf.get_variable('fc_weight', [2*hidden_size, 1],
                            initializer = fc_initializer)
# fc_weight: Storage for the fully connected layer's weights.
fc_bias = tf.get_variable('bias', [1])
# fc_bias: Storage for the fully connected layer's bias.

# tf.GraphKeys.REGULARIZATION_LOSSES:  A key to a collection in the graph
#   designated for losses due to regularization.
#   In this case, this portion of loss is regularization on the weights
#   for the fully connected layer.
tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(fc_weight))

x = tf.concat([hyp, evi], 1) # N, (Lh+Le), d
# Permuting batch_size and n_steps
x = tf.transpose(x, [1, 0, 2]) # (Le+Lh), N, d
# Reshaping to (n_steps*batch_size, n_input)
x = tf.reshape(x, [-1, vector_size]) # (Le+Lh)*N, d
# Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
x = tf.split(x, l_seq,)

# x: the inputs to the bidirectional_rnn


# tf.contrib.rnn.static_bidirectional_rnn: Runs the input through
#   two recurrent networks, one that runs the inputs forward and one
#   that runs the inputs in reversed order, combining the outputs.
rnn_outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm, lstm_back, x, dtype=tf.float32)
# rnn_outputs: the list of LSTM outputs, as a list.
#   What we want is the latest output, rnn_outputs[-1]

regression_scores = tf.matmul(rnn_outputs[-1], fc_weight) + fc_bias
# The scores are the regression


with tf.variable_scope('Accuracy'):
    predicts = tf.cast(regression_scores, 'int32')
    y_label = tf.cast(y, 'int32')
    corrects = tf.losses.mean_squared_error(predicts, y_label)
    num_corrects = tf.reduce_sum(tf.cast(corrects, tf.float32))
    accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))

with tf.variable_scope("loss"):
    loss = tf.losses.mean_squared_error(labels = y, predictions = regression_scores)
    total_loss = loss + weight_decay * tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

optimizer = tf.train.GradientDescentOptimizer(learning_rate)

opt_op = optimizer.minimize(total_loss)

# Initialize variables
init = tf.global_variables_initializer()

# Use TQDM if installed
tqdm_installed = False
try:
    from tqdm import tqdm

    tqdm_installed = True
except:
    pass

# Launch the Tensorflow session
sess = tf.Session()
sess.run(init)



# for 25 epochs
for epoch in range(0,25):
    print("--------- epoch ", str(epoch), " -----------")
    # training_iterations_count: The number of data pieces to train on in total
    # batch_size: The number of data pieces per batch
    training_iterations = range(0, training_iterations_count, batch_size)

    if tqdm_installed:
        # Add a progress bar if TQDM is installed
        training_iterations = tqdm(training_iterations)

    for i in training_iterations:
        # Select indices for a random data subset
        batch = np.random.randint(data_feature_list[0].shape[0], size=batch_size)

        # Use the selected subset indices to initialize the graph's
        #   placeholder values
        hyps, evis, ys = (data_feature_list[0][batch, :],
                          data_feature_list[1][batch, :],
                          correct_scores[batch])

        # Run the optimization with these initialized values
        sess.run([opt_op], feed_dict={hyp: hyps, evi: evis, y: ys})
        # display_step: how often the accuracy and loss should
        #   be tested and displayed.
        if (i / batch_size) % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={hyp: hyps, evi: evis, y: ys})
            # Calculate batch loss
            tmp_loss = sess.run(loss, feed_dict={hyp: hyps, evi: evis, y: ys})
            # Display results
            print("Iter " + str(i / batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(tmp_loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))


test_ids, test_feature_list, goldenTruth = split_data_into_scores(test_set_file)
test_hyps, test_evis = (test_feature_list[0], test_feature_list[1])

prediction = sess.run(regression_scores, feed_dict={hyp: test_hyps, evi: test_evis, y: [[0]]})
sess.close()

print(goldenTruth.tolist())
print(prediction.tolist())

print("Pearson correlation coefficient: ", str(pearsonr(goldenTruth, prediction)))

with open((output_dir+"Q2_results4.txt"), "w") as data:
    data.write("Pair_ID" + "\t" + "relatedness_score" + "\n")
    for index in range(0, len(goldenTruth)):
        data.write(str(test_ids[index])+ "\t" + str(prediction[index][0]) + "\n")