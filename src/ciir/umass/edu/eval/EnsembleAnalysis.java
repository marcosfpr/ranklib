package ciir.umass.edu.eval;

import java.util.ArrayList;
import java.util.List;

import ciir.umass.edu.features.FeatureManager;
import ciir.umass.edu.features.LinearNormalizer;
import ciir.umass.edu.features.SumNormalizor;
import ciir.umass.edu.learning.LinearRegRank;
import ciir.umass.edu.learning.RANKER_TYPE;
import ciir.umass.edu.learning.RankList;
import ciir.umass.edu.learning.Ranker;
import ciir.umass.edu.learning.RankerFactory;
import ciir.umass.edu.learning.boosting.AdaRank;
import ciir.umass.edu.learning.boosting.RankBoost;
import ciir.umass.edu.learning.ensemble.Ensemble;
import ciir.umass.edu.learning.neuralnet.ListNet;
import ciir.umass.edu.learning.neuralnet.RankNet;
import ciir.umass.edu.metric.MetricScorer;
import ciir.umass.edu.metric.NDCGScorer;
import ciir.umass.edu.metric.PrecisionScorer;
import ciir.umass.edu.metric.ReciprocalRankScorer;

public class EnsembleAnalysis {
    public static void main(String[] args) {
        String sampleFile = "/home/marcos/Projetos/.datasets/rotten/output/features/all_folds.txt";

        List<RankList> samples = FeatureManager.readInput(sampleFile);
        int[] features = FeatureManager.getFeatureFromSampleVector(samples);

        LinearNormalizer normalizer = new LinearNormalizer();
        normalizer.normalize(samples);

        List<RankList> trainingData = new ArrayList<RankList>();
        List<RankList> testData = new ArrayList<RankList>();

        FeatureManager.prepareSplit(samples, 0.8, trainingData, testData);

        MetricScorer trainScorer = new PrecisionScorer(2);

        RANKER_TYPE type = RANKER_TYPE.ENSEMBLE;

        // =========================================================================================
        // //
        // setup Rankers
        AdaRank ada = new AdaRank();
        AdaRank.nIteration = 500;
        AdaRank.maxSelCount = 10;
        AdaRank.tolerance = 0.0;
        ada.setMetricScorer(trainScorer);
        ada.setFeatures(features);
        ada.setTrainingSet(trainingData);

        RankBoost rb = new RankBoost();
        RankBoost.nThreshold = 10;
        RankBoost.nIteration = 500;
        rb.setMetricScorer(trainScorer);
        rb.setFeatures(features);
        rb.setTrainingSet(samples);

        RankNet rn = new RankNet();
        RankNet.learningRate = 0.0001;
        RankNet.nHiddenLayer = 2;
        RankNet.nHiddenNodePerLayer = 10;
        RankNet.nIteration = 10;
        rn.setMetricScorer(trainScorer);
        rn.setFeatures(features);
        rn.setTrainingSet(samples);

        ListNet ln = new ListNet();
        ListNet.learningRate = 0.0001;
        ListNet.nHiddenLayer = 2;
        ListNet.nHiddenNodePerLayer = 10;
        ListNet.nIteration = 50;
        ln.setMetricScorer(trainScorer);
        ln.setFeatures(features);
        ln.setTrainingSet(samples);

        LinearRegRank lr = new LinearRegRank();
        lr.setMetricScorer(trainScorer);
        lr.setFeatures(features);       
        lr.setTrainingSet(samples);    

        switch (type){
            case ADARANK:
                ada.init();
                ada.learn();

                ada.save("/home/marcos/Projetos/.pacotes/ranklib/modelos/adarank");

                System.out.printf("P@2 on test data: %.4f\n", evaluate(ada, testData, new PrecisionScorer(2)));
                System.out.printf("RR@2 on test data: %.4f\n", evaluate(ada, testData, new ReciprocalRankScorer(2)));
                System.out.printf("NDCG@2 on test data: %.4f\n", evaluate(ada, testData, new NDCGScorer(2)));
            break;
            case RANKBOOST:
                rb.init();
                rb.learn();

                rb.save("/home/marcos/Projetos/.pacotes/ranklib/modelos/rankboost");

                System.out.printf("P@2 on test data: %.4f\n", evaluate(rb, testData, new PrecisionScorer(2)));
                System.out.printf("RR@2 on test data: %.4f\n", evaluate(rb, testData, new ReciprocalRankScorer(2)));
                System.out.printf("NDCG@2 on test data: %.4f\n", evaluate(rb, testData, new NDCGScorer(2)));
            break;
            case RANKNET:
                rn.init();
                rn.learn();

                rn.save("/home/marcos/Projetos/.pacotes/ranklib/modelos/ranknet");

                System.out.printf("P@2 on test data: %.4f\n", evaluate(rn, testData, new PrecisionScorer(2)));
                System.out.printf("RR@2 on test data: %.4f\n", evaluate(rn, testData, new ReciprocalRankScorer(2)));
                System.out.printf("NDCG@2 on test data: %.4f\n", evaluate(rn, testData, new NDCGScorer(2)));
            break;
            case LISTNET:
                ln.init();
                ln.learn();

                ln.save("/home/marcos/Projetos/.pacotes/ranklib/modelos/listnet");

                System.out.printf("P@2 on test data: %.4f\n", evaluate(ln, testData, new PrecisionScorer(2)));
                System.out.printf("RR@2 on test data: %.4f\n", evaluate(ln, testData, new ReciprocalRankScorer(2)));
                System.out.printf("NDCG@2 on test data: %.4f\n", evaluate(ln, testData, new NDCGScorer(2)));
            break;
            case LINEAR_REGRESSION:
                lr.init();
                lr.learn();

                lr.save("/home/marcos/Projetos/.pacotes/ranklib/modelos/linearreg");

                System.out.printf("P@2 on test data: %.4f\n", evaluate(lr, testData, new PrecisionScorer(2)));
                System.out.printf("RR@2 on test data: %.4f\n", evaluate(lr, testData, new ReciprocalRankScorer(2)));
                System.out.printf("NDCG@2 on test data: %.4f\n", evaluate(lr, testData, new NDCGScorer(2)));

            break;
            case ENSEMBLE:
                RankerFactory rf = new RankerFactory();
                List<Ranker> rankers = new ArrayList<Ranker>();
                rankers.add(rf.loadRankerFromFile("/home/marcos/Projetos/.pacotes/ranklib/modelos/adarank"));
                rankers.add(rf.loadRankerFromFile("/home/marcos/Projetos/.pacotes/ranklib/modelos/rankboost"));
                rankers.add(rf.loadRankerFromFile("/home/marcos/Projetos/.pacotes/ranklib/modelos/ranknet"));
                rankers.add(rf.loadRankerFromFile("/home/marcos/Projetos/.pacotes/ranklib/modelos/listnet"));
                rankers.add(rf.loadRankerFromFile("/home/marcos/Projetos/.pacotes/ranklib/modelos/linearreg"));

                Ensemble en = new Ensemble(rankers);

                // en.init();
                // en.learn();

                System.out.printf("P@2 on test data: %.4f\n", evaluate(en, testData, new PrecisionScorer(2)));
                System.out.printf("RR@2 on test data: %.4f\n", evaluate(en, testData, new ReciprocalRankScorer(2)));
                System.out.printf("NDCG@2 on test data: %.4f\n", evaluate(en, testData, new NDCGScorer(2)));

            break;
            default:
                System.out.println("[Model Not Implemented]");
        }

    }
    
    private static double evaluate(Ranker ranker, List<RankList> rl, MetricScorer scorer)
	{
		List<RankList> l = rl;
		if(ranker != null)
			l = ranker.rank(rl);
		return scorer.score(l);
	}
}
