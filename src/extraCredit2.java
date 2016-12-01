import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;


public class extraCredit2 {
	
	public extraCredit2(String file){
		
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
			Instances testingInstances = new Instances(dataInstances, 0, count/5);
			testingInstances.setClassIndex(8);
			Instances learningInstances = new Instances(dataInstances, (count/5), 4*count/5);
			learningInstances.setClassIndex(8);
			Classifier classifier = new J48();
			classifier.buildClassifier(learningInstances);
			Evaluation evaluation = new Evaluation(learningInstances);
			evaluation.evaluateModel(classifier, testingInstances);
			String results = evaluation.toSummaryString();
			System.out.println(results);
			System.out.println(classifier.toString());
			
		} catch(MalformedURLException e){
			
		} catch(IOException e){
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}
	
	public static void main(String[] args) {
		new extraCredit2("diabetes.arff");

	}

}
