(ns mnist-clojure.core
  (:import
   [org.deeplearning4j.datasets DataSets]
   org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
   org.deeplearning4j.nn.conf.NeuralNetConfiguration$Builder
   org.deeplearning4j.nn.api.OptimizationAlgorithm
   org.deeplearning4j.nn.conf.Updater
   org.deeplearning4j.nn.conf.layers.DenseLayer$Builder
   org.deeplearning4j.nn.weights.WeightInit
   org.deeplearning4j.nn.conf.layers.OutputLayer$Builder
   org.nd4j.linalg.lossfunctions.LossFunctions$LossFunction
   org.deeplearning4j.nn.multilayer.MultiLayerNetwork
   org.deeplearning4j.optimize.listeners.ScoreIterationListener
   org.deeplearning4j.ui.api.UIServer
   org.deeplearning4j.ui.storage.InMemoryStatsStorage
   org.deeplearning4j.ui.stats.StatsListener
   org.deeplearning4j.api.storage.listener.RoutingIterationListener
   org.deeplearning4j.optimize.api.TrainingListener
   org.deeplearning4j.eval.Evaluation))


;(def train-data (DataSets/mnist))

(def numRows 28)
(def numColumns 28)
(def outputNum 10)
(def batchSize 128)
(def rndSeed 123) ;random seed for reproducibility
(def numEpochs 15)

(def train-iter (MnistDataSetIterator. batchSize true rndSeed))
(def test-iter (MnistDataSetIterator. batchSize true rndSeed))



;set up the config
(def conf
  (-> (NeuralNetConfiguration$Builder.)
      (.seed rndSeed)
      (.optimizationAlgo (OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT))
      (.iterations 1)
      (.learningRate 0.006)
      (-> (.updater Updater/NESTEROVS) (.momentum 0.9))
      (-> (.regularization true) (.l2 1e-4))
      (.list)
      (.layer 0 (-> (DenseLayer$Builder.)
                    (.nIn (* numRows numColumns))
                    (.nOut 1000)
                    (.activation "relu")
                    (.weightInit WeightInit/XAVIER)
                    (.build)))
      (.layer 1 (-> (OutputLayer$Builder. (LossFunctions$LossFunction/NEGATIVELOGLIKELIHOOD))
                    (.nIn 1000)
                    (.nOut outputNum)
                    (.activation "softmax")
                    (.weightInit WeightInit/XAVIER)
                    (.build)))
      (.pretrain false)
      (.backprop true)
      (.build)))

(def model
  (MultiLayerNetwork. conf))

(.init model)

(.getListeners model)


(def uiserver (UIServer/getInstance))

(def stats-storage  (InMemoryStatsStorage.))

(.setListeners model [(StatsListener. stats-storage)])

(.attach uiserver stats-storage)

(supers StatsListener)

(for [x (range 15)]
  (.fit model train-iter))


(def eval (Evaluation. outputNum))

(while (.hasNext test-iter)
  (let [next (.next test-iter)
        output (.output model (.getFeatureMatrix next))]
      (.eval eval (.getLabels next) output)))
