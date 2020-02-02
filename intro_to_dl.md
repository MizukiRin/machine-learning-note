# Introduction to Deep Learning

## Deep learning 3 steps:
Neural Network -> Goodness of func. -> Pick the best func.

### STEP1: Netural Network
-   **Neuron**: The node in the neural network->activation func
-   Different connection leads to different network
-   In the netural network we have a lot of neurons, which are  activation func(maybe logistic regression func), to connect, and each connection has it's weight and bias, which are network parameters called $\theta$.
-   We should design the connection way between each neuron.

#### Fully Connect Feedforward Network
-   **Fully**: all neurons in previous layer connect to all neurons in the next layer one by one
-   **Feedforward**: the signal flow in one direction, start in input layer, from previous layer to next layer, until to the output layer. There is not feedback between each layer.
![](.\images\intro_to_dl\1.png)
(横线上的是weight， 绿框框里是bias）

- A netural network with known weight and bias can seem as a function.
    - Input: vector
    - Output: vector

-   If we don't know the weights and bias, just know the structure of the netural network, that equal to we define a funciton set.
 -> **Given network structure, define a function set.**

**DEEP=Many Hidden Layers**

#### Neural Network:
Input layer -(W1,b1)-> hidden layer1 -(W2,b2)-> hidden layer3 -…(Wl, bl)-> output layer -> output
*Assume all the activation func are sigmoid func, input is x, x, W,b are all vectors*
![](.\images\intro_to_dl\2.png)
- Algorithm:
    `For j in (number of hidden layer + 1):`
        `y -> sigmoid(W[j]*x+b[j])`
        `x- >y`
    `END`
-   A series matrix product vector and add vector
-   Using parallel computing tech to speed up matrix operation, then you could call GPU to deal with it.
-   我们可以把整个neutral network看作如下通过hidden layer进行feature transform的结构：
![](.\images\intro_to_dl\3.png)
    -   multi-classclassifier要通过一个softmax func得到最后的输出  
    -   **Softmax**:歸一化指數函數。能將一個含任意實數的K維向量「壓縮」到另一個K維實向量中，使得每一個元素的範圍都在之間，並且所有元素的和為1(也可視為一個 (k-1)維的hyperplan，因為總和為1，所以是subspace)
        -   $\sigma(z)_{j} = \frac{e^{z_{j}}}{\sum^{K}_{k-1}e^{z_{k}}} \space\space _{for\space j=1,...,K}$



## STEP2: Goodness of func.
![](.\images\intro_to_dl\4.png)
-   调整参数，调整cross entropy（越小越好）
![](.\images\intro_to_dl\5.png)
-   由于你拥有大量的data， 所以我们要计算total loss。然后我们在function set里面找一组function他的total loss是minimize的， 或者找一组network的parameter可以minimizes total。 为了找出使得minimize成立的$\theta$， 我们采用gradient descent。那么，在neutral network中要如何用gradient descent的方法来train一个neutral network，这里我们就要采用backpropagation.

### Backpropagation
![](.\images\intro_to_dl\6.png)
我们先回顾一波gradient descent的运作方式。先找一个初始参数$\theta^{0}$，如何计算$\theta^{0}$对loss func的gradient， 即计算每一个network里面的参数（$w_{1},w_{2},b_{1},b_{2}$等等）对$L(\theta)$的偏微分，得到一个vector。然后我们可以更新参数，把$\theta^{0}$减掉learning rate， 如此反复。
**但是，在实际操作中，这个$\theta$ vector会是上百万维的，那要如何有效地去把这一个百万维的vector计算出来，这就是backpropagation要做的事情。Backpropagation是一个有效计算gradient descent的演算法。**

#### Chain Rule
两个case了解chain rule： 前面的变化会影响后面的， 即连锁影响。 backpropagation主要用到了chain rule.
![](.\images\intro_to_dl\7.png)

#### backpropagation
-   $Y^{n}$: output, $\hat{Y}^{n}$: target
-   Loss function = summation over all training data's loss value $C^{n}$, $C^{n}$ is the distance between $y^{n}$ and $\hat{y}^{n}$. 
    -   Do the partial differential of param $w$, and we get a new formula. And now we don't need to think about how to calculate $\frac{\partial L} {\partial w}$, we just focus on each data's $\frac{\partial C^{n}}{\partial w}$.
-   我们先只对一个neuron进行分析。整个梯度计算被分为forward pass和backward pass两部分。
![](.\images\intro_to_dl\8.png)

    - Forwardpass
        根据规律 就是每一阶段的input值，比如$\frac{\partial z}{\partial w_{1}} = x_{1}$, $\frac{\partial{z}}{\partial{w_{2}}}=x_{2}$，第一个layer计算出来的output就是第二个的input
    - Backward pass
        把他拆分为$\frac{\partial{a}}{\partial{z}} \frac{\partial{C}}{\partial{a}}$
        -   那么$\frac{\partial{a}}{\partial{z}}$的计算方式如图。我们可以看到这段就相当于z对activate func的偏微分。
![](.\images\intro_to_dl\9.png)
        -   对于$\frac{\partial{C}}{\partial{a}}$，我们利用chain rule对其继续分解，如图
![](.\images\intro_to_dl\10.png)
    		假设我们知道后面的$\partial{z'}$和$\partial{z''}$的值， 那么我们就可以顺利算出$\frac{\partial{C}}{\partial{a}}$，从而算出$\frac{\partial{C}}{\partial{z}}$
		    既然这段叫backward pass, 那我们换一种思考方式，就先假设从后面往前推导。想象有另外一个neuron如下，input在右边，output在左边，相当于一个放大器op-amp。因为我们之前以及计算好z了，所以$\sigma'(z)$就相当于一个常数。
        -   那么接下来的问题就是$\partial{z'}和\partial{z''}$的值要怎么去计算出来。那我们结合后面的layers来看，分为两个case进行讨论
            -   CASE1 Output Layer
![](.\images\intro_to_dl\11.png)
            即下一个layer就是output layer了。那根据chain rule，partial_{z'}和partial_{z''}就可以按照图上拆解方式得到
            -   CASE2 Not output layer
            Compute $\frac{\partial{C}}{\partial{z}}$ from the output layer.
            相当于，建一个反向的neural network
![](.\images\intro_to_dl\12.png)

## Summary
![](.\images\intro_to_dl\13.png)


>### Reference
> - Hung-yi Lee, *Machine Learning*, Brief Introduction of Deep Learning
> - Hung-yi Lee, *Machine Learning*, Backpropagation



