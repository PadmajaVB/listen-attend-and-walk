# listen-attend-and-walk
Code for converting natural language instructions into its corresponding action sequence. You can find my PyCon talk on this topic [here](https://www.youtube.com/watch?v=MJBWAkE7cEo).

We have followed **two different approches** to solve this problem

### First approach : Using Multi-level aligner based model

* This uses both high-level and low level representation of the input sentence to calculate the context vector. Refer [this](http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12522/12021) paper for the complete architecture. 

* You can find PyCon talk 

* PyTorch implementation of this approach can be found in **MultiLevelAlignerBasedModel** folder.

* Scratch code for this approach can be found in **CodeFromScratch** folder.

### Second approch : Using Global attention based model

* This only uses the high level representation of the input sentence to calculate the context vector. Refer [this](https://arxiv.org/pdf/1508.04025.pdf) paper for the complete architecture.

* PyTorch implementation of this approach can be found in **GlobalAttentionBasedModel** folder.

**AnotherApproach** folder contains keras implementation. This approach is still in progress, any sort of contribution in building the custom keras layer for multi-level aligner is most welcome.

## Contributers

This repository is created and maintained by

* [Padmaja Bhagwat](https://github.com/PadmajaVB)
* [Manisha Jhawar](https://github.com/ManishaJhawar)
* [Nitya C K](https://github.com/NityaCK)
