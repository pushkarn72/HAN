
import torch
import torch.nn as nn
from typing import Tuple

from sent_encoder import SentenceEncoder



        



class HAN(nn.Module):
    """
    Implementation of Hierarchial Attention Network (HAN) proposed in paper [1].

    Parameters
    ----------
    n_classes : int
        Number of classes

    vocab_size : int
        Number of words in the vocabulary

    embeddings : torch.Tensor
        Word embedding weights

    emb_size : int
        Size of word embeddings

    fine_tune : bool
        Allow fine-tuning of embedding layer? (only makes sense when using
        pre-trained embeddings)

    word_rnn_size : int
        Size of (bidirectional) word-level RNN

    sentence_rnn_size : int
        Size of (bidirectional) sentence-level RNN

    word_rnn_layers : int
        Number of layers in word-level RNN

    sentence_rnn_layers : int
        Number of layers in sentence-level RNN

    word_att_size : int
        Size of word-level attention layer

    sentence_att_size : int
        Size of sentence-level attention layer

    dropout : float, optional, default=0.5
        Dropout
    """
    def __init__(
        self,
        n_classes: int,
        vocab_size: int,
        embeddings: torch.Tensor,
        emb_size: int,
        fine_tune: bool,
        word_rnn_size: int,
        sentence_rnn_size: int,
        word_rnn_layers: int,
        sentence_rnn_layers: int,
        word_att_size: int,
        sentence_att_size: int,
        dropout: float = 0.5
    ) -> None:
        super(HAN, self).__init__()

        # sentence encoder
        # print("HAN init 0 ")
        self.sentence_encoder = SentenceEncoder(
            vocab_size, embeddings, emb_size, fine_tune,
            word_rnn_size, sentence_rnn_size,
            word_rnn_layers, sentence_rnn_layers,
            word_att_size, sentence_att_size,
            dropout
        )
    

        # classifier
        # print("HAN init 2 ")
        # self.fc = nn.Linear(2*sentence_rnn_size,100)
        # print("HAN init 3 ")
        self.fc_senti = nn.Linear(2*sentence_rnn_size,100)
        # print("HAN init 4 ")
        self.fc_compl = nn.Linear(2*sentence_rnn_size,100)
        # print("HAN init 5 ")
        self.fc_emoti = nn.Linear(2*sentence_rnn_size,100)
        # print("HAN init 6 ")
        self.fc_sever = nn.Linear(2*sentence_rnn_size,100)
        # print("HAN init 7 ")
        #self.fc_512 = nn.Linear(2 * sentence_rnn_size,512)
        
        #self.fc_256 = nn.Linear(512,256)
        #self.fc_100 = nn.Linear(256,100)
        # print("no of class  : ")
        # print(n_classes)
        # print("HAN init 8 ")
        # self.out = nn.Linear(100,n_classes)
        # print("HAN init 9 ")
        self.senti_out = nn.Linear(100,3)
        # print("HAN init 10 ")
        self.emoti_out=nn.Linear(100,7)
        # print("HAN init 11 ")
        self.compl_out = nn.Linear(100,2)  
        # print("HAN init 12 ")
        self.sever_out = nn.Linear(100,5)


        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        documents: torch.Tensor,
        sentences_per_document: torch.Tensor,
        words_per_sentence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        documents : torch.Tensor (n_documents, sent_pad_len, word_pad_len)
            Encoded document-level data

        sentences_per_document : torch.Tensor (n_documents)
            Document lengths

        words_per_sentence : torch.Tensor (n_documents, sent_pad_len)
            Sentence lengths

        Returns
        -------
        scores : torch.Tensor (batch_size, n_classes)
            Class scores

        word_alphas : torch.Tensor
            Attention weights on each word

        sentence_alphas : torch.Tensor
            Attention weights on each sentence
        """


        # sentence encoder, get document vectors
        # print("HAN forward 1 ")
        document_embeddings, word_alphas, sentence_alphas = self.sentence_encoder(
            documents,
            sentences_per_document,
            words_per_sentence
        )  # (n_documents, 2 * sentence_rnn_size), (n_documents, max(sentences_per_document), max(words_per_sentence)), (n_documents, max(sentences_per_document))
       
        dropout_embed= self.dropout(document_embeddings)
       
        scores_senti = nn.Tanh()(self.fc_senti(dropout_embed))  # (n_documents, n_classes)
        # print("scores_senti")
        # print(scores_senti)
        # print("HAN forward 4 ")
        scores_emoti = nn.Tanh()(self.fc_emoti(dropout_embed))  # (n_documents, n_classes)
        # print("scores_emoti")
        # print(scores_emoti)
        # print("HAN forward 5 ")
        scores_sever = nn.Tanh()(self.fc_sever(dropout_embed))  # (n_documents, n_classes)
        # print("scores_sever")
        # print(scores_sever)
        # print("HAN forward 6 ")
        scores_complain = nn.Tanh()(self.fc_compl(dropout_embed)) # (n_documents, n_classes)
        
        concat = torch.cat((scores_complain,scores_sever,scores_emoti,scores_senti),1)
        
    
        senti_out = self.senti_out(scores_senti)
        # print("HAN forward 10 ")
        emoti_out = self.emoti_out(scores_emoti)
        # print("HAN forward 11 ")
        sever_out = self.sever_out(scores_sever)
        # print("HAN forward 12 ")
        label_out = self.compl_out(scores_complain)

        return  emoti_out,senti_out,sever_out,label_out
       