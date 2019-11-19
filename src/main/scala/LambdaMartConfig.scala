/**
 * Configuration for Lambda MART algorithm
 *
 * @param trainPath Directory with train data
 * @param validatePath Directory with validation data
 * @param featuresColumn The name of the features column
 * @param labelColumn The name of the label column
 * @param evalAt Size of ranking evaluation
 * @param numIterations Number of iterations of the trainer
 * @param numLeaves Number of leaves in each tree
 * @param learningRate The learning rate of the training
 * @param earlyStoppingRound The early exit round
 * @param treeParallelism "Tree learner parallelism, can be set to 'data_parallel' or 'voting_parallel'
 * @param verbosity Logging level, < 0 is Fatal, eq 0 is Error, eq 1 is Info, > 1 is Debug
 * @param modelSavePath The path to save trained model
 */
case class LambdaMartConfig(trainPath: String, validatePath: String,
                            featuresColumn: String,
                            labelColumn: String,
                            evalAt: Int,
                            numIterations: Int,
                            numLeaves: Int,
                            learningRate: Double,
                            earlyStoppingRound: Int,
                            treeParallelism: String,
                            verbosity: Int,
                            modelSavePath: String)


