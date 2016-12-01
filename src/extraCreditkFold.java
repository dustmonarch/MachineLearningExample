import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;


public class extraCreditkFold {
	
	public extraCreditkFold(String file){
		
		try{
			ArffLoader arffloader = new ArffLoader();
			File arffFile = new File(file);
			arffloader.setFile(arffFile);
			Instances dataInstances = arffloader.getDataSet();
			int count = 0;
			while(count < dataInstances.numInstances()){
//				System.out.println("Instance: " + dataInstances.instance(count));
				count++;
			}
			dataInstances.randomize(new Random());
			dataInstances.setClassIndex(4);
			dataInstances.stratify(5);
			for(int n = 0; n < 5; n++){
				Instances train = dataInstances.trainCV(5, n);
				Instances test = dataInstances.testCV(5, n);
				Classifier classifier = new J48();
				classifier.buildClassifier(train);
				Evaluation evaluation = new Evaluation(train);
				evaluation.evaluateModel(classifier, test);
				String results = evaluation.toSummaryString();
//				System.out.println(results);
				System.out.println(classifier.toString());
			}
			
		} catch(MalformedURLException e){
			
		} catch(IOException e){
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}
	
	public static void main(String[] args) {
		new extraCreditkFold("iris.arff");

	}

}
