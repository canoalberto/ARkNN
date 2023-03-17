package moa.classifiers.multilabel;

import java.util.ArrayList;
import java.util.List;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import com.yahoo.labs.samoa.instances.MultiLabelInstance;
import com.yahoo.labs.samoa.instances.MultiLabelPrediction;
import com.yahoo.labs.samoa.instances.Prediction;

import moa.classifiers.AbstractMultiLabelLearner;
import moa.classifiers.MultiLabelClassifier;
import moa.core.Measurement;

public class ARkNN extends AbstractMultiLabelLearner implements MultiLabelClassifier {

	private static final long serialVersionUID = 1L;

	public IntOption STMmaxWindowSize = new IntOption("STMmaxWindowSize", 'w', "The maximum number of instances in the short-term memory", 500, 1, Integer.MAX_VALUE);
	
	public IntOption k = new IntOption("k", 'k', "The number of neighbors", 3, 1, Integer.MAX_VALUE);
	
	public FloatOption minimumFitnessOption = new FloatOption( "fitness", 'f', "The minimum fitness for an instance to stay alive", 0.001, 0, 1);

	private List<Instance> STMwindow;
	private double[] attributeRangeMin;
	private double[] attributeRangeMax;
	private double prequentialSuccesses;
	private double prequentialDenominator;
	
	@Override
	public String getPurposeString() {
		return "Multi-label Self-Adjusting Dual-Memory kNN for Drifting Data Streams";
	}

	@Override
	public void setModelContext(InstancesHeader context) {
		try {
			prequentialSuccesses = 0;
			prequentialDenominator = 0;
			
			STMwindow = new ArrayList<Instance>();
			attributeRangeMin = new double[context.numInputAttributes()];
			attributeRangeMax = new double[context.numInputAttributes()];

		} catch(Exception e) {
			System.err.println("Error: no model context available.");
			e.printStackTrace();
			System.exit(1);
		}
	}

	@Override
	public void resetLearningImpl() {
		prequentialSuccesses = 0;
		prequentialDenominator = 0;
		
		if(STMwindow != null)	STMwindow.clear();
	}

	@Override
	public void trainOnInstanceImpl(MultiLabelInstance inst) {
		
		updateRanges(inst);
		
		for(int i = 0; i < STMwindow.size(); i++) {
			STMwindow.get(i).setWeight(STMwindow.get(i).weight() * 0.995);
		}
		
		STMwindow.add(inst);
		
		for(int i = STMwindow.size() - 1; i >= 0; i--) {
        	if(STMwindow.get(i).weight() < minimumFitnessOption.getValue()) {
        		STMwindow.remove(i);
        	}
        }
		
		if(STMwindow.size() > STMmaxWindowSize.getValue()) {
			int worstInstanceIndex = -1;
        	double worsttInstance = Double.MAX_VALUE;
        	
        	for(int i = 0; i < STMwindow.size(); i++) {
            	if(STMwindow.get(i).weight() < worsttInstance) {
            		worstInstanceIndex = i;
            		worsttInstance = STMwindow.get(i).weight();
            	}
            }
        	
        	STMwindow.remove(worstInstanceIndex);
		}
	}

	/**
	 * Predicts the label of a given sample
	 */
	@Override
	public Prediction getPredictionForInstance(MultiLabelInstance instance) {

		MultiLabelPrediction prediction = new MultiLabelPrediction(instance.numberOutputTargets());

		double[] distancesToSTM = get1ToNDistances(instance, STMwindow);

		int[] nnIndicesSTM = nArgMin(Math.min(distancesToSTM.length, this.k.getValue()), distancesToSTM);
		int[] successfulpredictionsSTM = new int[nnIndicesSTM.length];
		double[] distances = new double[nnIndicesSTM.length];
		
		for (int i = 0; i < nnIndicesSTM.length; i++) {
			distances[i] = 2.0 - distancesToSTM[nnIndicesSTM[i]];
		}
		
		for(int j = 0; j < instance.numberOutputTargets(); j++) {

			double votesPositive = 0;
			double votesNegative = 0;
			
			for (int i = 0; i < nnIndicesSTM.length; i++) {
				int nnIdx = nnIndicesSTM[i];
				
				if (STMwindow.get(nnIdx).classValue(j) == 1)
					votesPositive += STMwindow.get(nnIdx).weight() * distances[i];
				else
					votesNegative += STMwindow.get(nnIdx).weight() * distances[i];
				
				if(STMwindow.get(nnIdx).classValue(j) == instance.classValue(j)) {
					successfulpredictionsSTM[i]++;
				}
			}
			
			double sum = votesPositive + votesNegative;
					
			votesPositive /= sum;
			votesNegative /= sum;
			
			prediction.setVotes(j, new double[]{votesNegative, votesPositive});
		}
		
		// Instance was cited, reset age based on current age and accuracy
		for (int i = 0; i < nnIndicesSTM.length; i++) {
			int nnIdx = nnIndicesSTM[i];
			double relativeSuccessRatio = (successfulpredictionsSTM[i] / instance.numberOutputTargets()) - prequentialSuccesses / prequentialDenominator;
			STMwindow.get(nnIdx).setWeight(Math.max(0, Math.min(1, STMwindow.get(nnIdx).weight() + relativeSuccessRatio)));
		}
		
		// Update prequential accuracy of dataset
		int successfulPredictions = 0;
		for(int j = 0; j < instance.numberOutputTargets(); j++) {
			int predictedLabel = prediction.getVotes(j)[1] > prediction.getVotes(j)[0] ? 1 : 0;
			if(instance.classValue(j) == predictedLabel)
				successfulPredictions++;
		}
		
		prequentialSuccesses = prequentialSuccesses * 0.995 + successfulPredictions / instance.numberOutputTargets();
		prequentialDenominator = prequentialDenominator * 0.995 + 1.0;
		
		return prediction;
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
	
	private double[] get1ToNDistances(Instance sample, List<Instance> samples) {

		double[] distances = new double[samples.size()];
		
		for (int i = 0; i < samples.size(); i++)
			distances[i] = VectorOperators.getCosAtt(samples.get(i), sample, this.attributeRangeMax, this.attributeRangeMin);
		
		return distances;
	}

	/**
	 * Returns the n smallest indices of the smallest values (sorted).
	 */
	private int[] nArgMin(int n, double[] values, int startIdx, int endIdx) {

		int indices[] = new int[n];

		for (int i = 0; i < n; i++) {
			double minValue = Double.MAX_VALUE;
			for (int j = startIdx; j < endIdx + 1; j++) {

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

	public int[] nArgMin(int n, double[] values) {
		return nArgMin(n, values, 0, values.length-1);
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