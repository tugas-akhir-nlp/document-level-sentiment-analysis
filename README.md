# document-level-sentiment-analysis

The code were used to build model for document level sentiment analysis. The model was built using several deep neural network topologies and document representation methods. The experiment used two different Indonesian text corpus, TripAdvisor (from baseline) and prosa.ai. Due to confidential issue, some resources such as corpus, word embedding, document embedding, and the model itself are not published in the repository.

## main.ipynb

Each model can be built by running the code in a notebook file named 'main.ipynb' located in src folder. The notebook is divided into three sections: 
1. To build document embedding model using paragraph vector 
2. To build sentiment analysis model using deep neural network
3. To do model evaluation using testing data

Each section can be run independently by import all modules and libraries in the first cell of the notebook.

#### A. Build Document Representation Model
In this section you can input a corpus name ('prosa' or 'tripadvisor') then the PvModel wil set up file path to save the model produced after learning. You will also need to input the file path of the corpus. If you want to build model for different corpus, you can directly modify the file path in PvModel.

#### B. Build Sentiment Analysis Model
In this section you can input or modify several variables based on deep neural network that you want to train. For all DNN you can change the 'embedding_size' based on word embedding that you use. You can also change the number of maximum vocabulary (max_vocab) and maximum sequence in a sentence (max_sequence) based on corpus.
##### 1. Convolutional Neural Network (Conv1D)
If you want to create CNN with single filter or kernel, set the 'extra_conv' variable into False and if you want to use several filters then set it into True. You also need to input the filter size in 'cnn_kernel'. If you want to add paragraph vector as input features of CNN, set 'paragraph_vec' into True and set 'doc_vector' into 'prosa' or 'tripadvisor' to import the corresponding document embedding.
##### 2. Recurrent Neural Network (Bi-LSTM and Bi-GRU)
You need to set 'rnn_type' into 'bi-lstm' or 'bi-gru' according to the model you want to build. Set the number of RNN hidden size in 'rnn_unit'. RNN model can also be built using paragraph vector. If you want to use paragraph vector as input features of RNN, set 'paragraph_vec' into True and set 'doc_vector' into 'prosa' or 'tripadvisor' to import the corresponding document embedding.
##### 3. Hierarchical Deep Neural Network (HDNN)
To create a hierarchical model, set 'hierarchy' into True. Then define the type of DNN used for each level. For sentence level, 'dnn_sent_level' can be set into 'cnn' or 'lstm'. For document level, 'dnn_doc_level' can be set into 'bi-lstm' or 'bi-gru'. The number of rnn unit for document level can be set in 'grnn_unit'. Corpus prosa.ai has compatible format of data to build the hierarchical model.
#### C. Sentiment Analysis Model Evaluation
In this section you need to set several variables according to the sentiment analysis model you load. You also need to specify the model path.
