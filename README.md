# Neural Network using C++
[![Build Status](https://travis-ci.com/glitch401/NeuralNet_cpp.svg?branch=master)](https://travis-ci.com/glitch401/NeuralNet_cpp)

While there are plethora of frameworks fit to every trivial learning needs to powering sateof the art algorithms.
This mini project is aimed at understanding the nitty-gritty details involved in powering the `Neural Networks`.

>While working on fake news detection, I saw a potential to develope a custom loss function.
>Which provoked me to take up this mini project.

This is an extension and based on David Miller's tutorial [Neural Net in C++ Tutorial](https://vimeo.com/19569529).

#### Abouts and Running
The sample example demonstrates an `XOR` operation, being learned by the `Neural Network`, implementaion demonstrates Net recent average error after forward and back propogation :
| In 0 | In 1 | Out |
| ------ | ------ |------ |
| 0 | 0 |0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 |0 |

`trainingData.txt` consists of `20,000` training samples by default--you could create greater number of trainig samples if required using:
```sh
g++ src/make_xor_training_data.cpp -o make_xor_training_data.out
./make_xor_training_data.out 10000
```
where the argument after `./make_xor_training_data.out` is the number of training samples, you might leave it blank for 20,000 training samples.

For demonstration of the `Neural Net` over the generated training samples:
```sh
g++ src/nn_cpp.cpp -o nn_cpp.out
./nn_cpp.out
```
or you might consider logging the training to a file:
```sh
./nn_cpp.out>out.txt
```
Last obtained `Net recent average error` : ` 0.0018979` on pass 20,000

#### ToDo
- enable usage of custom `NN topology`--current is set to default of `2 4 1`
- enable training across other datasets following test and validations too.

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)


   [nn_Cpp]: <https://github.com/glitch401/NeuralNet_cpp>
   [git-repo-url]: <https://github.com/glitch401/NeuralNet_cpp.git>
   [indranil biswas]: <http://glitch401.github.io>

