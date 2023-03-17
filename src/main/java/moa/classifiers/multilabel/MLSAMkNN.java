/*
 *  Modified from https://github.com/Waikato/moa/blob/master/moa/src/main/java/moa/classifiers/lazy/SAMkNN.java
 *
 */

package moa.classifiers.multilabel;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.*;
import moa.classifiers.AbstractMultiLabelLearner;
import moa.classifiers.MultiLabelClassifier;
import moa.clusterers.kmeanspm.CoresetKMeans;
import moa.core.Measurement;

import java.util.*;

public class MLSAMkNN extends AbstractMultiLabelLearner implements MultiLabelClassifier {

	private static final long serialVersionUID = 1L;

	public IntOption kOption = new IntOption("k", 'k', "The number of neighbors", 5, 1, Integer.MAX_VALUE);

	public IntOption limitOption = new IntOption("limit", 'w', "The maximum number of instances to store", 1000, 1, Integer.MAX_VALUE);

	public IntOption minSTMSizeOption = new IntOption("minSTMSize", 'm', "The minimum number of instances in the STM", 50, 1, Integer.MAX_VALUE);

	public FloatOption relativeLTMSizeOption = new FloatOption("relativeLTMSize", 'p', "The allowed LTM size relative to the total limit.", 0.4, 0.0, 1.0);

	public FlagOption debugMode = new FlagOption("debug", 'g', "Debug");

	private String[] metrics = {"Subset Accuracy", "Hamming Score", "Accuracy", "Precision", "Recall", "F-measure"};

	public MultiChoiceOption metric = new MultiChoiceOption("metric", 'e', "Choose metric used to adjust memory", metrics, metrics, 1);

	@Override
	public String getPurposeString() {
		return "Multi-label, self adjusting memory KNN";
	}

	//to keep things straight
	private int reunion = 0;
	private int interse = 1;
	private int onestrue = 2;
	private int onespred = 3;
	private int hamming = 4;

	private int numLabels;
	private Instances stm;
	private Instances ltm;
	private int maxLTMSize;
	private int maxSTMSize;
	private List<int[]> stmHistory;
	private List<int[]> ltmHistory;
	private List<int[]> cmHistory;
	private double[][] distanceMatrixSTM;
	private Map<Integer, List<int[]>> predictionHistories;
	private double[] attributeRangeMin;
	private double[] attributeRangeMax;
	private Random random;

	@Override
	public void setModelContext(InstancesHeader context) {
		try {
			this.numLabels = context.numOutputAttributes();
			this.stm = new Instances(context,0);
			this.stm.setClassIndex(context.classIndex());
			this.ltm = new Instances(context,0);
			this.ltm.setClassIndex(context.classIndex());
			this.attributeRangeMin = new double[context.numInputAttributes()];
			this.attributeRangeMax = new double[context.numInputAttributes()];
			this.distanceMatrixSTM = new double[limitOption.getValue()+1][limitOption.getValue()+1];
			this.predictionHistories = new HashMap<>();
			this.maxLTMSize = (int)(relativeLTMSizeOption.getValue() * limitOption.getValue());
			this.maxSTMSize = limitOption.getValue() - this.maxLTMSize;
			this.stmHistory = new ArrayList<>();
			this.ltmHistory = new ArrayList<>();
			this.cmHistory = new ArrayList<>();
			this.random = new Random();
		} catch(Exception e) {
			System.err.println("Error: no Model Context available.");
			e.printStackTrace();
			System.exit(1);
		}
	}

	@Override
	public void resetLearningImpl() {
		if(this.stm != null)
		{
			this.stm.delete();
			this.ltm.delete();
			this.stmHistory.clear();
			this.ltmHistory.clear();
			this.cmHistory.clear();
			this.distanceMatrixSTM = new double[limitOption.getValue()+1][limitOption.getValue()+1];
			this.predictionHistories = new HashMap<>();
		}
	}

	@Override
	public void trainOnInstanceImpl(MultiLabelInstance inst) {

		/*
		 * some print statements for testing
		 */
		if(debugMode.isSet())
		{
			System.out.print("Instances in STM:");
			for(int l = 0; l < this.stm.size(); l++){
				System.out.print("(");
				System.out.print(this.stm.get(l));
				System.out.print(")    ");
			}
			System.out.println();
			System.out.print("Instances in LTM:");
			for(int l = 0; l < this.ltm.size(); l++){
				System.out.print("(");
				System.out.print(this.ltm.get(l));
				System.out.print(")    ");
			}
			System.out.println();
			System.out.println();

			System.out.print("STM growth: " + stm.size());
		}
		/*
		 * end print statements
		 */

		this.stm.add(inst);

		updateRanges(inst);

		/*
		 * more print statements
		 */
		if(debugMode.isSet())
			System.out.println(" -> " + stm.size());
		/*
		 * end print statements
		 */

		memorySizeCheck();

		if(debugMode.isSet())
			System.out.println("Should call clean last: clean against is STM, to clean is LTM");

		clean(this.stm, this.ltm, true);

		double distancesSTM[] = this.get1ToNDistances(inst, this.stm);

		for (int i =0; i < this.stm.numInstances(); i++)
			this.distanceMatrixSTM[this.stm.numInstances()-1][i] = distancesSTM[i];

		int oldWindowSize = this.stm.numInstances();
		int newWindowSize = this.getNewSTMSize();

		/*
		 * more print statements
		 */
		if(debugMode.isSet())
			System.out.println("Old window size: " + oldWindowSize + "    New window size: " + newWindowSize);
		/*
		 * end print statements
		 */

		if (newWindowSize < oldWindowSize) {
			int diff = oldWindowSize - newWindowSize;
			Instances discardedSTMInstances = new Instances(this.stm, 0);

			for (int i = diff; i>0; i--) {
				discardedSTMInstances.add(this.stm.get(0).copy());
				this.stm.delete(0);
			}

			for (int i = 0; i < this.stm.numInstances(); i++)
				for (int j = 0; j < this.stm.numInstances(); j++)
					this.distanceMatrixSTM[i][j] = this.distanceMatrixSTM[diff+i][diff+j];

			for (int i = 0; i < diff; i++) {
				if(this.ltmHistory.size() == this.stmHistory.size() && this.ltmHistory.size() > 0) //don't remove from LTM until it catches up with the STM
					this.ltmHistory.remove(0);
				if(this.stmHistory.size() > 0)
					this.stmHistory.remove(0);
				if(this.cmHistory.size() > 0)
					this.cmHistory.remove(0);
			}

			if(debugMode.isSet())
				System.out.println("Should call clean all: cleanAgainst is STM, to clean is discarded.");

			this.clean(this.stm, discardedSTMInstances, false);

			for (int i = 0; i < discardedSTMInstances.numInstances(); i++)
				this.ltm.add(discardedSTMInstances.get(i).copy());

			memorySizeCheck();
		}

		/*
		 * more print statements
		 */
		if(debugMode.isSet())
		{
			System.out.print("Instances in STM:");
			for(int l = 0; l < this.stm.size(); l++){
				System.out.print("(");
				System.out.print(this.stm.get(l));
				System.out.print(")    ");
			}
			System.out.println();
			System.out.print("Instances in LTM:");
			for(int l = 0; l < this.ltm.size(); l++){
				System.out.print("(");
				System.out.print(this.ltm.get(l));
				System.out.print(")    ");
			}
			System.out.println();
		}
		/*
		 * end print statements
		 */
	}

	/**
	 * Predicts the label of a given sample by using the STM, LTM and the CM.
	 */
	@Override
	public Prediction getPredictionForInstance(MultiLabelInstance instance) {

		MultiLabelPrediction pSTM = new MultiLabelPrediction(instance.numberOutputTargets());
		MultiLabelPrediction pLTM = new MultiLabelPrediction(instance.numberOutputTargets());
		MultiLabelPrediction pCM = new MultiLabelPrediction(instance.numberOutputTargets());
		MultiLabelPrediction p = new MultiLabelPrediction(instance.numberOutputTargets());
		double distancesSTM[];
		double distancesLTM[];

		if (this.stm != null && this.stm.numInstances() > 0) {
			distancesSTM = get1ToNDistances(instance, this.stm);
			int nnIndicesSTM[] = nArgMin(Math.min(distancesSTM.length, this.kOption.getValue()), distancesSTM);
			pSTM = getPrediction(nnIndicesSTM, this.stm);

			if (this.ltm.numInstances() > 0) {

				distancesLTM = get1ToNDistances(instance, this.ltm);
				pCM = getCMPrediction(distancesSTM, this.stm, distancesLTM, this.ltm);
				int nnIndicesLTM[] = nArgMin(Math.min(distancesLTM.length, this.kOption.getValue()), distancesLTM);
				pLTM = getPrediction(nnIndicesLTM, this.ltm);

				/*
				 * more print statements
				 */
				if(debugMode.isSet())
					System.out.println("Predictions: STM -> " + pSTM + "\tLTM -> " + pLTM + "\tCM -> " + pCM);
				/*
				 * end print statements
				 */

				double metricSTM = getMetricFromHistory(this.stmHistory);
				double metricLTM = getMetricFromHistory(this.ltmHistory);
				double metricCM = getMetricFromHistory(this.cmHistory);

				if(debugMode.isSet())	System.out.println();
				if(debugMode.isSet())	System.out.println("STM metric: " + metricSTM + "\t LTM metric: " + metricLTM + "\t CM metric: " + metricCM);
				if (metricSTM >= metricLTM && metricSTM >= metricCM) {
					if(debugMode.isSet())	System.out.println("STM prediction used.");
					p = pSTM;
				} else if (metricLTM > metricSTM && metricLTM >= metricCM) {
					if(debugMode.isSet())	System.out.println("LTM prediction used.");
					p = pLTM;
				} else {
					if(debugMode.isSet())	System.out.println("CM prediction used.");
					p = pCM;
				}
			} else {
				if(debugMode.isSet())	System.out.println("STM prediction used.");
				p = pSTM;
				pCM = pSTM;
			}

			stmHistory.add(getMetricSums(instance, pSTM));
			cmHistory.add(getMetricSums(instance, pCM));
			if (this.ltm != null && this.ltm.numInstances() > 0)
				ltmHistory.add(getMetricSums(instance, pLTM));
		}

		/*
		 * more print statements
		 */
		if(debugMode.isSet())
		{
			System.out.println("STM History: ");
			for(int[] pred_array : stmHistory) {
				System.out.print(Arrays.toString(pred_array));
			}
			System.out.println();

			System.out.println("LTM History: ");
			for(int[] pred_array : ltmHistory) {
				System.out.print(Arrays.toString(pred_array));
			}
			System.out.println();

			System.out.println("CM History: ");
			for(int[] pred_array : cmHistory) {
				System.out.print(Arrays.toString(pred_array));
			}
			System.out.println();
		}
		/*
		 * end print statements
		 */

		return p;
	}

	/**
	 * Returns the votes for each label.
	 */
	private MultiLabelPrediction getPrediction(int[] nnIndices, Instances instances){

		MultiLabelPrediction prediction = new MultiLabelPrediction(this.numLabels);

		for(int j = 0; j < numLabels; j++)
		{
			int count = 0;

			for (int nnIdx : nnIndices)
				if(instances.instance(nnIdx).classValue(j) == 1)
					count++;

			double relativeFrequency = count / (double) nnIndices.length;

			prediction.setVotes(j, new double[]{1.0 - relativeFrequency, relativeFrequency});
		}

		return prediction;
	}

	/**
	 * Returns the distance weighted votes for the combined memory (CM).
	 */
	private MultiLabelPrediction getCMPrediction(double distancesSTM[], Instances stm, double distancesLTM[], Instances ltm){
		double[] distancesCM = new double[distancesSTM.length + distancesLTM.length];
		System.arraycopy(distancesSTM, 0, distancesCM, 0, distancesSTM.length);
		System.arraycopy(distancesLTM, 0, distancesCM, distancesSTM.length, distancesLTM.length);
		int nnIndicesCM[] = nArgMin(Math.min(distancesCM.length, this.kOption.getValue()), distancesCM);
		return getPredictionCM(nnIndicesCM, stm, ltm);
	}

	private MultiLabelPrediction getPredictionCM(int[] nnIndices, Instances stm, Instances ltm){

		MultiLabelPrediction prediction = new MultiLabelPrediction(this.numLabels);

		for(int j = 0; j < numLabels; j++)
		{
			int count = 0;

			for (int nnIdx : nnIndices) {
				if (nnIdx < stm.numInstances()) {
					if (stm.instance(nnIdx).classValue(j) == 1)
						count++;
				}
				else {
					if (ltm.instance(nnIdx-stm.numInstances()).classValue(j) == 1)
						count++;
				}
			}

			double relativeFrequency = count / (double) nnIndices.length;

			prediction.setVotes(j, new double[]{1.0 - relativeFrequency, relativeFrequency});

		}

		return prediction;
	}

	private int[] getMetricSums(Instance instance, MultiLabelPrediction prediction){

		int sumReunion = 0;
		int sumInterse = 0;
		int sumOnesTrue = 0;
		int sumOnesPred = 0;
		int correct = 0;

		int[] metrics = new int[5];

		/** preset threshold */
		double t = 0.5;

		for (int j = 0; j < prediction.numOutputAttributes(); j++) {

			int yp = (prediction.getVote(j, 1) >= t) ? 1 : 0;
			int y_true = (int) instance.valueOutputAttribute(j);

			correct += ((int) instance.classValue(j) == yp) ? 1 : 0;

			if (y_true == 1 || yp == 1)
				sumReunion++;

			if (y_true == 1 && yp == 1)
				sumInterse++;

			if (y_true == 1)
				sumOnesTrue++;

			if (yp == 1)
				sumOnesPred++;
		}

		metrics[reunion] = sumReunion;
		metrics[interse] = sumInterse;
		metrics[onestrue] = sumOnesTrue;
		metrics[onespred] = sumOnesPred;
		metrics[hamming] = correct;

		return metrics;
	}

	private double getMetricFromHistory(List<int[]> history){

		double sumSubsetAccuracy = 0.0;
		double sumHamming = 0.0;
		double sumAccuracy = 0.0;
		double sumPrecision = 0.0;
		double sumRecall = 0.0;
		double sumFmeasure = 0.0;

		int sumExamples = 0;

		for(int[] instanceSum : history) {

			sumExamples++;

			//Accuracy by instance(Jaccard Index)
			if (instanceSum[reunion] > 0)
				sumAccuracy += (double) instanceSum[interse] / instanceSum[reunion];
			else
				sumAccuracy += 0.0;

			//Precision by instance
			if (instanceSum[onestrue] > 0)
				sumPrecision += (double) instanceSum[interse] / instanceSum[onestrue];

			//Recall by instance
			if (instanceSum[onespred] > 0)
				sumRecall += instanceSum[interse] / (double) instanceSum[onespred];

			//F-Measure by instance
			if ((instanceSum[onespred] + instanceSum[onestrue]) > 0)
				sumFmeasure += (double) 2 * instanceSum[interse] / (instanceSum[onespred] + instanceSum[onestrue]);
			else
				sumFmeasure += 0.0;

			sumHamming += (instanceSum[hamming] / (double) numLabels);            // Hamming Score
			sumSubsetAccuracy += (instanceSum[hamming] == numLabels) ? 1 : 0;        // Exact Match
		}

		String metric = this.metric.getChosenLabel();
		switch(metric) {
		case "Subset Accuracy" : return sumSubsetAccuracy/sumExamples;
		case "Hamming Score" : return sumHamming/sumExamples;
		case "Accuracy" : return sumAccuracy/sumExamples;
		case "Precision" : return sumPrecision/sumExamples;
		case "Recall" : return sumRecall/sumExamples;
		case "F-measure" : return sumFmeasure/sumExamples;
		default : return sumHamming/sumExamples;
		}
	}

	/**
	 * Returns the n smallest indices of the smallest values (sorted).
	 */
	private int[] nArgMin(int n, double[] values, int startIdx, int endIdx){

		int indices[] = new int[n];

		for (int i = 0; i < n; i++){
			double minValue = Double.MAX_VALUE;
			for (int j = startIdx; j < endIdx + 1; j++){

				if (values[j] < minValue){
					boolean alreadyUsed = false;
					for (int k = 0; k < i; k++){
						if (indices[k] == j){
							alreadyUsed = true;
						}
					}
					if (!alreadyUsed){
						indices[i] = j;
						minValue = values[j];
					}
				}
			}
		}
		return indices;
	}

	public int[] nArgMin(int n, double[] values){
		return nArgMin(n, values, 0, values.length-1);
	}

	/**
	 * Returns the Euclidean distance between one sample and a collection of samples in an 1D-array.
	 */
	private double[] get1ToNDistances(Instance sample, Instances samples) {

		double distances[] = new double[samples.numInstances()];

		for (int i = 0; i < samples.numInstances(); i++)
			distances[i] = this.getDistance(sample, samples.get(i));

		return distances;
	}

	/**
	 * Returns the Euclidean distance.
	 */
	private double getDistance(Instance instance1, Instance instance2) {

		//		REGULAR DISTANCE
		//        for (int i = 0; i < instance1.numInputAttributes(); i++)
		//            distance += (instance1.valueInputAttribute(i) - instance2.valueInputAttribute(i)) * (instance1.valueInputAttribute(i) - instance2.valueInputAttribute(i));

		//		NORMALIZED DISTANCE
		//        for(int i = 0; i < instance1.numInputAttributes(); i++)
		//        {
		//            double val1 = instance1.valueInputAttribute(i);
		//            double val2 = instance2.valueInputAttribute(i);
		//
		//            if(attributeRangeMax[i] - attributeRangeMin[i] != 0)
		//            {
		//                val1 = (val1 - attributeRangeMin[i]) / (attributeRangeMax[i] - attributeRangeMin[i]);
		//                val2 = (val2 - attributeRangeMin[i]) / (attributeRangeMax[i] - attributeRangeMin[i]);
		//                distance += (val1 - val2) * (val1 - val2);
		//            }
		//        }

		//		int firstI = -1, secondI = -1;
		//		int firstNumValues  = instance1.numValues();
		//		int secondNumValues = instance2.numValues();
		//		int numAttributes   = instance1.numAttributes();
		//		int numOutputs      = instance1.numOutputAttributes();
		//
		//		for (int p1 = 0, p2 = 0; p1 < firstNumValues || p2 < secondNumValues;) {
		//
		//			if (p1 >= firstNumValues) {
		//				firstI = numAttributes;
		//			} else {
		//				firstI = instance1.index(p1);
		//			}
		//
		//			if (p2 >= secondNumValues) {
		//				secondI = numAttributes;
		//			} else {
		//				secondI = instance2.index(p2);
		//			}
		//
		//			if (firstI < numOutputs) {
		//				p1++;
		//				continue;
		//			}
		//
		//			if (secondI < numOutputs) {
		//				p2++;
		//				continue;
		//			}
		//
		//			if (firstI == secondI) {
		//				int idx = firstI - numOutputs;
		//				if(attributeRangeMax[idx] - attributeRangeMin[idx] != 0)
		//				{
		//					double val1 = instance1.valueSparse(p1);
		//					double val2 = instance2.valueSparse(p2);
		//					val1 = (val1 - attributeRangeMin[idx]) / (attributeRangeMax[idx] - attributeRangeMin[idx]);
		//					val2 = (val2 - attributeRangeMin[idx]) / (attributeRangeMax[idx] - attributeRangeMin[idx]);
		//					distance += (val1 - val2) * (val1 - val2);
		//				}
		//				p1++;
		//				p2++;
		//			} else if (firstI > secondI) {
		//				int idx = secondI - numOutputs;
		//				if(attributeRangeMax[idx] - attributeRangeMin[idx] != 0)
		//				{
		//					double val2 = instance2.valueSparse(p2);
		//					val2 = (val2 - attributeRangeMin[idx]) / (attributeRangeMax[idx] - attributeRangeMin[idx]);
		//					distance += (val2) * (val2);
		//				}
		//				p2++;
		//			} else {
		//				int idx = firstI - numOutputs;
		//				if(attributeRangeMax[idx] - attributeRangeMin[idx] != 0)
		//				{
		//					double val1 = instance1.valueSparse(p1);
		//					val1 = (val1 - attributeRangeMin[idx]) / (attributeRangeMax[idx] - attributeRangeMin[idx]);
		//					distance += (val1) * (val1);
		//				}
		//				p1++;
		//			}
		//		}

		double distance = 0;

		if(instance1.numValues() == instance1.numAttributes()) // Dense Instance
		{
			for(int i = 0; i < instance1.numInputAttributes(); i++)
			{
				double val1 = instance1.valueInputAttribute(i);
				double val2 = instance2.valueInputAttribute(i);

				if(attributeRangeMax[i] - attributeRangeMin[i] != 0)
				{
					val1 = (val1 - attributeRangeMin[i]) / (attributeRangeMax[i] - attributeRangeMin[i]);
					val2 = (val2 - attributeRangeMin[i]) / (attributeRangeMax[i] - attributeRangeMin[i]);
					distance += (val1 - val2) * (val1 - val2);
				}
			}
		}
		else // Sparse Instance
		{
			int firstI = -1, secondI = -1;
			int firstNumValues  = instance1.numValues();
			int secondNumValues = instance2.numValues();
			int numAttributes   = instance1.numAttributes();
			int numOutputs      = instance1.numOutputAttributes();

			for (int p1 = 0, p2 = 0; p1 < firstNumValues || p2 < secondNumValues;) {

				if (p1 >= firstNumValues) {
					firstI = numAttributes;
				} else {
					firstI = instance1.index(p1);
				}

				if (p2 >= secondNumValues) {
					secondI = numAttributes;
				} else {
					secondI = instance2.index(p2);
				}

				if (firstI < numOutputs) {
					p1++;
					continue;
				}

				if (secondI < numOutputs) {
					p2++;
					continue;
				}

				if (firstI == secondI) {
					int idx = firstI - numOutputs;
					if(attributeRangeMax[idx] - attributeRangeMin[idx] != 0)
					{
						double val1 = instance1.valueSparse(p1);
						double val2 = instance2.valueSparse(p2);
						val1 = (val1 - attributeRangeMin[idx]) / (attributeRangeMax[idx] - attributeRangeMin[idx]);
						val2 = (val2 - attributeRangeMin[idx]) / (attributeRangeMax[idx] - attributeRangeMin[idx]);
						distance += (val1 - val2) * (val1 - val2);
					}
					p1++;
					p2++;
				} else if (firstI > secondI) {
					int idx = secondI - numOutputs;
					if(attributeRangeMax[idx] - attributeRangeMin[idx] != 0)
					{
						double val2 = instance2.valueSparse(p2);
						val2 = (val2 - attributeRangeMin[idx]) / (attributeRangeMax[idx] - attributeRangeMin[idx]);
						distance += (val2) * (val2);
					}
					p2++;
				} else {
					int idx = firstI - numOutputs;
					if(attributeRangeMax[idx] - attributeRangeMin[idx] != 0)
					{
						double val1 = instance1.valueSparse(p1);
						val1 = (val1 - attributeRangeMin[idx]) / (attributeRangeMax[idx] - attributeRangeMin[idx]);
						distance += (val1) * (val1);
					}
					p1++;
				}
			}
		}

		return Math.sqrt(distance);
	}

	private void updateRanges(MultiLabelInstance instance) {
		for(int i = 0; i < instance.numInputAttributes(); i++)
		{
			if(instance.valueInputAttribute(i) < attributeRangeMin[i])
				attributeRangeMin[i] = instance.valueInputAttribute(i);
			if(instance.valueInputAttribute(i) > attributeRangeMax[i])
				attributeRangeMax[i] = instance.valueInputAttribute(i);
		}
	}

	/*
	 * Makes sure that the STM and LTM combined doe not surpass the maximum size.
	 */
//	private void memorySizeCheck(){
//		if (this.stm.numInstances() + this.ltm.numInstances() > this.maxSTMSize + this.maxLTMSize){
//			if (this.ltm.numInstances() > this.maxLTMSize)
//				this.clusterDown();
//
//			else { //shift values from STM directly to LTM since STM is full
//
//				/*
//				 * more print statements
//				 */
//				if(debugMode.isSet())
//					System.out.println("Shift to LTM");
//				/*
//				 * end print statements
//				 */
//
//				int numShifts = this.maxLTMSize - this.ltm.numInstances() + 1;
//				for (int i = 0; i < numShifts; i++){
//					this.ltm.add(this.stm.get(0).copy());
//					this.stm.delete(0);
//					if(this.ltmHistory.size() == this.stmHistory.size() && this.ltmHistory.size() > 0) //don't remove from LTM until it catches up with the STM
//						this.ltmHistory.remove(0);
//					if(this.stmHistory.size() > 0)
//						this.stmHistory.remove(0);
//					if(this.cmHistory.size() > 0)
//						this.cmHistory.remove(0);
//				}
//
//				this.clusterDown();
//				this.predictionHistories.clear();
//				for (int i = 0; i < this.stm.numInstances(); i++)
//					for (int j = 0; j < this.stm.numInstances(); j++)
//						this.distanceMatrixSTM[i][j] = this.distanceMatrixSTM[numShifts+i][numShifts+j];
//			}
//		}
//	}
	
	/*
	 * Makes sure that the STM and LTM combined does not surpass the maximum size. Best fix, I think.
	 */
	private void memorySizeCheck(){
		if(debugMode.isSet()) {
			System.out.println("Checking memory size.");
			System.out.println("STM size: " + this.stm.numInstances() + " Max is: " + this.maxSTMSize);
			System.out.println("LTM size: " + this.ltm.numInstances() + " Max is: " + this.maxLTMSize);
		}

		// First check, and attempt to reduce LTM size
		if (this.stm.numInstances() + this.ltm.numInstances() > this.maxSTMSize + this.maxLTMSize)
			if (this.ltm.numInstances() > this.maxLTMSize)
				this.clusterDown();

		// Second check, move from STM to LTM, then attempt to reduce LTM size
		if (this.stm.numInstances() + this.ltm.numInstances() > this.maxSTMSize + this.maxLTMSize){

			//check if LTM is larger than max size (which can happen...)
			int numShifts = 1;
			if (this.ltm.numInstances() < this.maxLTMSize)
				numShifts = this.maxLTMSize - this.ltm.numInstances() + 1;

			for (int i = 0; i < numShifts; i++){
				this.ltm.add(this.stm.get(0).copy());
				this.stm.delete(0);
				if(this.ltmHistory.size() == this.stmHistory.size() && this.ltmHistory.size() > 0) //don't remove from LTM until it catches up with the STM
					this.ltmHistory.remove(0);
				if(this.stmHistory.size() > 0)
					this.stmHistory.remove(0);
				if(this.cmHistory.size() > 0)
					this.cmHistory.remove(0);
			}

			this.clusterDown();
			this.predictionHistories.clear();
			for (int i = 0; i < this.stm.numInstances(); i++)
				for (int j = 0; j < this.stm.numInstances(); j++)
					this.distanceMatrixSTM[i][j] = this.distanceMatrixSTM[numShifts+i][numShifts+j];

		}
	}

	/**
	 * Performs classwise kMeans++ clustering for given samples with corresponding labels. The number of samples is halved per class.
	 */
	private void clusterDown(){

		//original SAMkNN clusters by class value. For multi-label, this is clustering by unique label set
		//THIS IS REALLY PROBLEMATIC IF THERE ARE TOO MANY LABELS!!!

		/*
		 * more print statements
		 */
		if(debugMode.isSet())
		{
			System.out.println("Clustering Down start - instances in LTM = " + this.ltm.size());
			for(int l = 0; l < this.ltm.size(); l++){
				System.out.print("(");
				System.out.print(this.ltm.get(l));
				System.out.print(")    ");
			}
			System.out.println();
		}
		/*
		 * end print statements
		 */

		//get a set of all the label sets present in the LTM
		List<int[]> label_sets = new ArrayList<>();
		for(int i = 0; i < this.ltm.numInstances(); i++) {
			int[] label_set = new int[numLabels];
			for(int j = 0; j < numLabels; j++)
				label_set[j] = (int) ltm.get(i).classValue(j);

			boolean duplicate = false;
			for(int[] set: label_sets)
				if(Arrays.equals(set,label_set))
					duplicate = true;

			if(!duplicate)
				label_sets.add(label_set);
		}

		/*
		 * more print statements
		 */
		if(debugMode.isSet())
		{
			System.out.println();
			System.out.println("Label sets:");
			for(int[] set: label_sets)
				System.out.print(Arrays.toString(set));
			System.out.println();
		}
		/*
		 * end print statements
		 */

		for(int[] set: label_sets) {  //for each unique label set in the LTM

			List<double[]> samplesWithSet = new ArrayList<>();

			for(int i = this.ltm.numInstances()-1; i >= 0; i--) { //run though all the instances in the LTM

				int[] label_set = new int[numLabels];  //get label set for the instance
				for(int j = 0; j < numLabels; j++)
					label_set[j] = (int) ltm.get(i).classValue(j);

				if(Arrays.equals(set,label_set)) { //if it equals the set we're working on
					samplesWithSet.add(ltm.get(i).toDoubleArray()); //add that instance to the list
					this.ltm.delete(i); //and delete it from the LTM
				}
			}

			if(debugMode.isSet())
				System.out.println("Set: " + Arrays.toString(set) + "    Number of Instances to Cluster: " + samplesWithSet.size());

			//kmeans++ expects a weight in the first index, weight all the same by overwriting first label with 1
			//could duplicate arrays to remove indices used by the labels, instead overwriting with 0, which shouldn't impact clustering
			for (double[] sample : samplesWithSet) {
				sample[0] = 1;
				for(int i = 1; i < numLabels; i++)
					sample[i] = 0;
			}

			List<double[]> centroids = this.kMeans(samplesWithSet, Math.max(samplesWithSet.size() / 2, 1));

			for (double[] centroid : centroids) {

				double[] instance_array = new double[this.ltm.numAttributes()];
				//returned centroids do not contain the weight anymore, but simply the data
				System.arraycopy(centroid, 0, instance_array, 1, this.ltm.numAttributes() - 1);

				for(int i = 0; i < numLabels; i++)
					instance_array[i] = set[i];

				Instance inst = new InstanceImpl(1, instance_array);
				inst.setDataset(this.ltm);
				this.ltm.add(inst);
			}

		}

		/*
		 * more print statements
		 */
		if(debugMode.isSet())
		{
			System.out.println("Clustering Down end - instances in LTM = " + this.ltm.size());
			for(int l = 0; l < this.ltm.size(); l++){
				System.out.print("(");
				System.out.print(this.ltm.get(l));
				System.out.print(")    ");
			}
			System.out.println();
		}
		/*
		 * end print statements
		 */
	}

	private List<double[]> kMeans(List<double[]> points, int k){

		List<double[]> centroids = CoresetKMeans.generatekMeansPlusPlusCentroids(k,points, this.random);
		CoresetKMeans.kMeans(centroids, points);
		return centroids;
	}

	/**
	 * Returns the bisected STM size which maximized the metric
	 */
	private int getNewSTMSize(){
		if(debugMode.isSet())	System.out.println();
		if(debugMode.isSet())	System.out.println("Calculating new window size.");

		int numSamples = this.stm.numInstances();
		if (numSamples < 2 * this.minSTMSizeOption.getValue())
			return numSamples;
		else {
			List<Integer> numSamplesRange = new ArrayList<>();
			numSamplesRange.add(numSamples);
			while (numSamplesRange.get(numSamplesRange.size() - 1) >= 2 * this.minSTMSizeOption.getValue())
				numSamplesRange.add(numSamplesRange.get(numSamplesRange.size() - 1) / 2);

			Iterator<Integer> it = this.predictionHistories.keySet().iterator();
			while (it.hasNext()) {
				Integer key = (Integer) it.next();
				if (!numSamplesRange.contains(numSamples - key))
					it.remove();
			}

			List<Double> metricList = new ArrayList<>();
			for (Integer numSamplesIt : numSamplesRange) {
				int idx = numSamples - numSamplesIt;
				List<int[]> predHistory;
				if (this.predictionHistories.containsKey(idx))
					predHistory = this.getIncrementalTestTrainPredHistory(this.stm, idx, this.predictionHistories.get(idx));
				else
					predHistory = this.getTestTrainPredHistory(this.stm, idx);

				this.predictionHistories.put(idx, predHistory);

				/*
				 * more print statements
				 */
				if(debugMode.isSet())
					System.out.println("\tMetric value starting at index " + idx + ": " + this.getMetricFromHistory(predHistory));
				/*
				 * end print statements
				 */

				metricList.add(this.getMetricFromHistory(predHistory));
			}
			int maxMetricIdx = metricList.indexOf(Collections.max(metricList));
			int windowSize = numSamplesRange.get(maxMetricIdx);
			if (windowSize < numSamples)
			{
				if(debugMode.isSet())
					System.out.println("Should adapt histories");
				this.adaptHistories(maxMetricIdx);
			}

			return windowSize;
		}
	}

	/**
	 * Creates a prediction history from the scratch.
	 */
	private List<int[]> getTestTrainPredHistory(Instances instances, int startIdx){

		/*
		 * more print statements
		 */
		if(debugMode.isSet())
			System.out.println("\tCreating prediction history from scratch.");
		/*
		 * end print statements
		 */

		List<int[]> predictionHistory = new ArrayList<>();

		for (int i = startIdx; i < instances.numInstances(); i++){

			int nnIndices[] = nArgMin(Math.min(this.kOption.getValue(), i - startIdx), distanceMatrixSTM[i], startIdx, i-1);
			MultiLabelPrediction prediction = getPrediction(nnIndices, instances);

			predictionHistory.add(getMetricSums(instances.get(i),prediction));
		}

		/*
		 * more print statements
		 */
		if(debugMode.isSet())
		{
			System.out.println();
			System.out.print("New prediction history for index " + startIdx + ": ");
			for(int[] pred_array : predictionHistory)
				System.out.print(Arrays.toString(pred_array));
			System.out.println();
		}
		/*
		 * end print statements
		 */

		return predictionHistory;
	}

	/**
	 * Creates a prediction history incrementally by using the previous predictions.
	 */
	private List<int[]> getIncrementalTestTrainPredHistory(Instances instances, int startIdx, List<int[]> predictionHistory){

		/*
		 * more print statements
		 */
		if(debugMode.isSet())
		{
			System.out.println("\tCreating prediction history incrementally.");
			System.out.print("\tStored prediction history for index " + startIdx + ": ");
			for(int[] pred_array : predictionHistory)
				System.out.print(Arrays.toString(pred_array));

			System.out.println();
		}
		/*
		 * end print statements
		 */

		for (int i = startIdx + predictionHistory.size(); i < instances.numInstances(); i++){
			int nnIndices[] = nArgMin(Math.min(this.kOption.getValue(), distanceMatrixSTM[i].length), distanceMatrixSTM[i], startIdx, i-1);
			MultiLabelPrediction prediction = getPrediction(nnIndices, instances);
			predictionHistory.add(getMetricSums(instances.get(i),prediction));
		}

		/*
		 * more print statements
		 */
		if(debugMode.isSet())
		{
			System.out.println();
			System.out.print("New prediction history for index " + startIdx + ": ");
			for(int[] pred_array : predictionHistory)
				System.out.print(Arrays.toString(pred_array));
			System.out.println();
		}
		/*
		 * end print statements
		 */

		return predictionHistory;
	}

	/**
	 * Removes predictions of the largest window size and shifts the remaining ones accordingly.
	 */
	private void adaptHistories(int numberOfDeletions){
		if(debugMode.isSet())
			System.out.println("Adapting histories");
		for (int i = 0; i < numberOfDeletions; i++){
			SortedSet<Integer> keys = new TreeSet<>(this.predictionHistories.keySet());
			this.predictionHistories.remove(keys.first());
			keys = new TreeSet<>(this.predictionHistories.keySet());
			for (Integer key : keys){
				List<int[]> predHistory = this.predictionHistories.remove(key);
				this.predictionHistories.put(key-keys.first(), predHistory);
			}
		}
	}

	private void cleanSingle(Instances cleanAgainst, int cleanAgainstindex, Instances toClean){
		Instances cleanAgainstTmp = new Instances(cleanAgainst);
		cleanAgainstTmp.delete(cleanAgainstindex);
		double distancesSTM[] = get1ToNDistances(cleanAgainst.get(cleanAgainstindex), cleanAgainstTmp);
		int nnIndicesSTM[] = nArgMin(Math.min(this.kOption.getValue(), distancesSTM.length), distancesSTM);

		double distancesLTM[] = get1ToNDistances(cleanAgainst.get(cleanAgainstindex), toClean);
		int nnIndicesLTM[] = nArgMin(Math.min(this.kOption.getValue(), distancesLTM.length), distancesLTM);
		double[] distThreshold = new double[numLabels];
		for (int i = 0; i < numLabels; i++) {
			distThreshold[i] = -1;
		}

		if(debugMode.isSet())
			System.out.println("Clean against: " + cleanAgainst.get(cleanAgainstindex));

		for (int nnIdx: nnIndicesSTM) {
			if(debugMode.isSet())
				System.out.println("Nearest Neighbor: " + cleanAgainstTmp.get(nnIdx));
			for (int j = 0; j < numLabels; j++) {
				if (cleanAgainstTmp.get(nnIdx).classValue(j) == cleanAgainst.get(cleanAgainstindex).classValue(j))
					if (distancesSTM[nnIdx] > distThreshold[j])
						distThreshold[j] = distancesSTM[nnIdx];
				if(debugMode.isSet())
					System.out.println("\tLabel: " + j + "\tThreshold: " + distThreshold[j]);
			}
		}

		List<Integer> delIndices = new ArrayList<>();
		for (int nnIdx: nnIndicesLTM) {
			boolean clean = false;
			for(int j = 0; j < numLabels; j++) {
				if (toClean.get(nnIdx).classValue() != cleanAgainst.get(cleanAgainstindex).classValue()) {
					if (distancesLTM[nnIdx] <= distThreshold[j]) {
						clean = true;
					}
				}
			}
			if(debugMode.isSet())	System.out.println("Instance: " + toClean.get(nnIdx) + "\tDistance: " + distancesLTM[nnIdx] + "\tClean: " + clean);
			if(clean)
				delIndices.add(nnIdx);
		}
		Collections.sort(delIndices, Collections.reverseOrder());
		for (Integer idx : delIndices)
			toClean.delete(idx);
	}
	/**
	 * Removes distance-based all instances from the input samples that contradict those in the STM.
	 */
	private void clean(Instances cleanAgainst, Instances toClean, boolean onlyLast) {
		if (cleanAgainst.numInstances() > this.kOption.getValue() && toClean.numInstances() > 0){
			if (onlyLast) {
				if(debugMode.isSet())	System.out.println("Cleaning last");
				cleanSingle(cleanAgainst, (cleanAgainst.numInstances() - 1), toClean);
			}
			else {
				if(debugMode.isSet())	System.out.println("Cleaning all");
				for (int i = 0; i < cleanAgainst.numInstances(); i++)
					cleanSingle(cleanAgainst, i, toClean);
			}
		}
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		return null;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
	}

	public boolean isRandomizable() {
		return false;
	}
}