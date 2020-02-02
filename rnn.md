# Recurrent Neural Network

## Eample Application
### Slot Filling
- E.g. ticket booking system, the system should know the info in the sentence people talk belong to which slot automatic.
    -   For example, when u tell the system "I would like to arrive Taipei on Nov 2nd", the system should know "Taipei" this word belongs to the slot "Destination", and "Nov 2nd" belongs to "Time". And other vocabulary do not belong to any slot.
-   Solving slot filling by Feedforward network
    -   INPUT: a word (each word is represented as a vector)
        -   1-of-N encoding
        The vector is lexicon size. Each dimension corresponds to a word in the lexicon. The dimension for the word is 1, and others are 0.
            -   Sometimes we will meet some vocabulary we don't know, so we should add a dimension for "Others" and collect the words which didn't appear in the dictionary. 
            -   Or you can use **Word Hashing**. Use n-gram letters to express the vector, you won't face to the problem that word do not in dictinary. For example, word "apple", it has "app","ppl" and "ple", so the n-gram letter "app", "ppl" and "ple" in dimension are 1, and others are 0. 
        -   Word vector
    -   OUTPUT: **Probability distribution** that the input word belonging to the slots.
    -   But there is a problem. There are two sentences, A is "I would like to arrive Taipei on Nov 2nd" and B is "I would like to leave Taipei on Nov 2nd". "Taipei" in A is destination, while in B is departure. So that is, whether our neural network can have memory that it has read the word "arrive" or "leave" before it read the word "Taipei". In other words, we hope the system can contextual understand words or sentences. And RNN(Recurrent Neural Network) is the way to solve this problem.
		
		
		
## Recurrent Neural Network (simple)
### Basic Concept
-   The OUTPUT of **hidden layer** are stored in the memory.  When it turns to next layer, memory can be considered as another input.
-   EXAMPLE: Assume that all the weights are 1, and no bias. All activation functions are linear.
![](.\images\rnn\1.png)
    -   Before using RNN, you should give the memory a initial values. And in this example, the initial values are 0.
    -   In first input [1,1], the memory will store [2,2] , and we get the output [4,4]
    -   In second input[1,1], the memory will store [6,6], and we get the output [12,12]
    -   In third input [2,2], the memory will store [16,16] and we get the output [32,32]
    -   We get the final output sequence is [4,4][12,12][32,32] 
-   **Changing the sequence order will change the output.** In RNN, it will consider the order of input sequence.
-   And let's see how RNN work on slot filling:
![](.\images\rnn\2.png)
    **The same network is used again and again**
    - Back to the question about "leave Taipei" and "arrive Taipei", we can know that the memory in the network when the sequence move to the word "Taipei" show different vector. Because the output of hidden layer is different, the final output will be different.
![](.\images\rnn\3.png)
-   RNN also can be deep, it can have many levels of hidden layer, and each hidden layers'output will be stored in the corresponding memory.

### Elman Network & Jordan Network
-   Elman Network: 把每个hidden layer的output存起来，在下个时间点再读出来（就是前面讲的）
-   Jordan Network: 存的是整个network的output值，整个output的值在下个时间点再读进来
![](.\images\rnn\4.png)
-   Jordan Network相比于Elman Network好的是， 它是有target的。也就是说，我们可以清楚我们放在memory里的是什么东西。而在Elman Network中，我们放在memory里的相当于是hidden info，我们不知道机器它算出来了什么东西，那这就是比较难控制的。

### Bidirectional RNN
-   你可以同时trian一个正向的RNN和一个逆向的RNN，把这两个RNN的hidden layer拿出来都接给一个output layer。
![](.\images\rnn\5.png)
-   Advantages: 产生output时看的范围是比较广的，真正联系了上下文。

## Long Short-term Memory(LSTM)
![](.\images\rnn\6.png)
-   THREE GATES:
    -   Input Gate: 决定其他neuron能否把值**输进去**
    -   Output Gate: 决定其他neuron能否把值**读出来**
    -   Forget Gate: 决定什么时候把存在memory里面的值format掉
    -   三个Gate的行为都是由机器自行学到
-   FOUR INPUT:
    -   想要被存到memory cell里面的值
    -   操控input gate的讯号
    -   操控output gate的讯号
    -   操控forget gate的讯号
-   ONE OUTPUT:
    -   其他neuron想要读取的值

### 一张图详解LSTM
假设在输入这些input values之前，我们的memory cell已经存了一个数值c。然后$z,\space z_{i},\space z_{f},\space z_{o}$，分别对应四个input， output是a这样子。
![](.\images\rnn\7.png)
-   $\space z_{i},\space z_{f},\space z_{o}$ 通过的这三个activation function $f$ 通常是sigmoid function。因为sigmoid func的值介于0和1之间，这个值表示gate被打开的程度。值为1是指被打开，值为0是关闭，mimic open and close gate。
-   再强调一遍sigmoid func的公式: $S(t)=\frac{1}{1+e^{-t}}$, t大于0，趋近于1，t小于0趋近于0.
-   我们将input的值$z$通过一个activate func得到$g(z)$,操控input gate的signal $z_{i}通过一个sigmoid fun得到$f(z_{i})$. 这两者做个相乘等到$g(z)f(z_{i})$。再来看操控forget gate这边，我们同样通过sigmoid func得到$f(z_{f})$,然后因为要考虑到已存在memory的值，我们就把两项相乘得到$cf(z_{f})，再返回cell里$。*(initial c value can be zero)* 所以$c'=g(z)f(z_{i})+cf(z_{f})$, $c'$是新存进memory cell的值。
-   forget gate打开时，即$f(z_{f})=1$代表记得，反之代表遗忘。forget gate平常都是打开的，input和output gate通常是关闭的。
-   然后将$c'$通过一个function得到$h(c')$。接下来就是看output gate让不让他通过这样子，也就是最后再做个乘积，我们的output $a=h(c')f(z_{o})$。

### EXAMPLE for LSTM
首先先看看我们要做一个怎样的LSTM,要求如图
![](.\images\rnn\8.png)
(蓝色是memory，红色是output)
我们把它放到图示上来计算看看，他是怎么运作的：
-   input的weight和bias是通过training data和gradient descent计算到的
-   我们先假设$g$和$h$都是linear，初始的memory是0
![](.\images\rnn\9.png)
-   input的地方我们设置的weight和bias是[1,0,0,0]，这表示我们输入$x_{1}$。在input gate是[0,100,0,-10],当$x_{2}$没有任何输入，即为0时，我们整个讯号的值时-10，这个情况下得到的sigmoid func的值是趋近于0。当$x_{2}$输入为-1时同样。而当$x_{2}$输入为1时乘上weight 100，sigmoid的值趋近为1。Output gate也一样，但是针对的时$x_{3}$. 这边要注意的是forget gate，因为他是常打开的，也就是要记住，所以我们bias设置为10，只有$x_{2}$更负的时候，sigmoid的值才会趋近于0，忘记memory。
-   可以根据图示上的输入sequence来计算看看。

### Compare to Original Network
-   在原始的neuron network中，有很多neuron。我们会把input乘上不同weight当作是不同的neuron的输入，每个neuron都是一个funciton，输入一个scalar，output另一个scalar。
-   And for LSTM, just simply replace the neurons with LSTM.
-   原来的neuron是一个input一个output，而LSTM是4个**不同的**input一个output。当同一个系统，你用LSTM的数目和neuron的数目是一样的时候，LSTM需要的参数量是一般neural network的四倍。

### WHY LSTM IS RNN
-   在时间点t，我们的input $x^{t}$（一个vector），通过4个transform（乘上一个matrix）得到vector $z,\space z_{i},\space z_{f},\space z_{o}$。那以$z$为例，$z$的每一个dimension就代表了操控每个LSTM的input。dimension数目就是LSTM cell的数目。
-   扔到一个cell里进行运作的 $z,\space z_{i},\space z_{f},\space z_{o}$ 都只是vector里的一个dimension。下图就是个simplified的LSTM。
![](.\images\rnn\10.png)
-   真正的LSTM如下所示：
![](.\images\rnn\11.png)
-   他会把hidden layer的输出把它接进来，当作下一个时间点的input。也就是说，不仅看当前的input还会看之前的output。
-   还会再加一个叫**peephole**的东西。这个东西把存在memory的值也拉到input这边。所以你在操控LSTM的四个gate的时候，你是同时考虑了x，h和c。

### Multiple-layer LSTM
![](.\images\rnn\12.png)


## HOW RNN LEARNING?
Example: 假设我们现在在做slot filling。我们trian一个sentence， 比如"arrive Taipei on Nov 2nd"。 

### Learning Target
我们知道要做learnig的话，要定义一个**cost func**来evaluate你的model的parameter好还是不好，然后pick一个model parameter可以让这个lost最小。
-   那在RNN中**cost**怎么定呢？把"arrive"扔到这个RNN中，最后得出$y^{1}$。接下来，这个$y^{1}$会和一个Reference的vector算它的cross entropy。也就是说，我们把"arrive"丢进去，我们希望那个$y^{1}$的reference vector应该是slot "other"的dimension是1，其他是0.那这个reference vector的长度就是你slot的数目。
    -   因为语境的缘故，在做training的时候，你不能把你utterance里面的word sequence打散来看，而是应该当作一个整体。这样子trian出来的才符合联系上下文得出的结果。
    -   你的cost就是每个时间点的RNN的output跟reference vector的cross entropy的和，也就是你要去minimize的对象。

### Learning
有了lost func， 来看training怎么做。
-   我们现在已经定出lost func $L$， 那我们要update这个network里面某一个参数$w$, 就是计算$\frac{\partial{L}}{\partial{w}}$,然后也是使用**gradient descent**去update每一个参数。
-   Backpropagation through time(BPTT)
    Back propagation的进阶版。因为RNN是在time sequence上运作
