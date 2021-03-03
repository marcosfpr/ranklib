package ciir.umass.edu.learning.ensemble;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import ciir.umass.edu.features.LinearNormalizer;
import ciir.umass.edu.learning.DataPoint;
import ciir.umass.edu.learning.RankList;
import ciir.umass.edu.learning.Ranker;
import ciir.umass.edu.utilities.MergeSorter;

/**
 * Ensemble method LTR
 */
public class Ensemble extends Ranker{
    private List<Ranker> rankers;

    public Ensemble(List<Ranker> rankers){
        this.rankers = rankers;
    }

    @Override
    public void init() {
        for(Ranker r : rankers){
            r.init();
        }

    }

    @Override
	public RankList rank(RankList rl)
	{   

        Map<DataPoint, Double> scoreDocs = new HashMap<DataPoint, Double>();
        for(int i = 0; i < rl.size(); i++){
            scoreDocs.put(rl.get(i), 0.0);
        }
        
        double[] normalizeFactors = new double[rankers.size()];

        for(int k = 0; k < rankers.size(); k++){
            Double sum = 0.0;

            for(int i = 0; i < rl.size(); i++){
                sum += rankers.get(k).eval(rl.get(i));
            }
            
            normalizeFactors[k] = sum;
        }

        for(int i = 0; i < rl.size(); i++){
            double score = 0.0; 
            
            for(int k = 0; k < rankers.size(); k++){
                score += rankers.get(k).eval(rl.get(i)) / normalizeFactors[k];
            }

            score /= rankers.size();

            scoreDocs.put(rl.get(i), score);
        }

        Map<DataPoint, Double> sortedDocs = scoreDocs.entrySet().stream()
                            .sorted(Entry.comparingByValue(Comparator.reverseOrder()))
                            .collect(Collectors.toMap(Entry::getKey, Entry::getValue, (e1, e2) -> e1, LinkedHashMap::new));
        
        List<DataPoint> rank = new ArrayList<DataPoint>();
        for(Map.Entry<DataPoint, Double> entry : sortedDocs.entrySet())
            rank.add(entry.getKey());

        return new RankList(rank);
    }

    @Override
    public void learn() {
        for(Ranker r : rankers){
            r.learn();
        }

    }

    @Override
    public Ranker createNew() {
        return new Ensemble(rankers);
    }

    @Override
    public String toString() {
        StringBuilder output = new StringBuilder();
        output.append("[+] Ensemble Method");

        for(Ranker r : rankers){
            output.append(r.toString());
        }

		return output.toString();
    }

    @Override
    public String model() {
        return toString();
    }

    @Override
    public void loadFromString(String fullText) {
        PRINTLN("NOT IMPLEMENTED.");
    }

    @Override
    public String name() {
        return "Ensemble";
    }

    @Override
    public void printParameters() {
        for(Ranker r : rankers){
            PRINT(r.name() + " parameters : ");
            r.printParameters();
        }

    }

}
