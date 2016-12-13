(ns mnist-clojure.core
  (:require [taoensso.timbre :as logger])
  (:import
   [org.deeplearning4j.datasets DataSets]
   org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
   org.deeplearning4j.nn.conf.NeuralNetConfiguration$Builder
   org.deeplearning4j.nn.api.OptimizationAlgorithm
   org.deeplearning4j.nn.conf.Updater
   ))

;(def train-data (DataSets/mnist))

(def numRows 28)
(def numColumns 28)
(def outputNum 10)
(def batchSize 128)
(def rndSeed 123) ;random seed for reproducibility
(def numEpochs 15)

(def train-iter (MnistDataSetIterator. batchSize true rndSeed))
(def test-iter (MnistDataSetIterator. batchSize true rndSeed))


(logger/info "Build model...")

;set up the config
(def conf
  (-> (NeuralNetConfiguration$Builder.)
      (.seed rndSeed)
      (.optimizationAlgo (OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT))
      (.iterations 1)
      (.learningRate 0.006)
      (-> (.updater Updater/NESTEROVS))))


 .seed(rngSeed) //include a random seed for reproducibility
                // use stochastic gradient descent as an optimization algorithm
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .learningRate(0.006) //specify the learning rate
                .updater(Updater.NESTEROVS).momentum(0.9) //specify the rate of change of the learning rate.
                .regularization(true).l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder() //create the first, input layer with xavier initialization
                        .nIn(numRows * numColumns)
                        .nOut(1000)
                        .activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
                        .nIn(1000)
                        .nOut(outputNum)
                        .activation("softmax")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .pretrain(false).backprop(true) //use backpropagation to adjust weights
                .build();

