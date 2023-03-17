package moa.classifiers.multilabel;

import java.util.Random;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import com.yahoo.labs.samoa.instances.MultiLabelInstance;
import com.yahoo.labs.samoa.instances.MultiLabelPrediction;
import com.yahoo.labs.samoa.instances.Prediction;

import moa.classifiers.AbstractMultiLabelLearner;
import moa.classifiers.MultiLabelClassifier;
import moa.core.Measurement;

public class MLkNN extends AbstractMultiLabelLearner implements MultiLabelClassifier  {

    private static final long serialVersionUID = 1L;

    public IntOption kOption = new IntOption( "k", 'k', "The number of neighbors", 10, 1, Integer.MAX_VALUE);

    public IntOption limitOption = new IntOption( "limit", 'w', "The maximum number of instances to store", 1000, 1, Integer.MAX_VALUE);

    @Override
    public String getPurposeString() {
        return "Multi-label KNN";
    }

    protected Instances window;

    private double[] priorCounts; 	// count of training instances with label
    private double[] priorProb; 	// probability that training instance has label
    private double[][] postProbOne;
    private double[][] postProbZero;

    // CANO ADD
    private double[] attributeRangeMin;
    private double[] attributeRangeMax;

    private int smooth = 1;

    @Override
    public void setModelContext(InstancesHeader context) {
        try {
            this.window = new Instances(context,0);
            this.window.setClassIndex(context.classIndex());
            this.attributeRangeMin = new double[context.numInputAttributes()];
            this.attributeRangeMax = new double[context.numInputAttributes()];
            this.priorCounts = new double[context.numOutputAttributes()];
            this.priorProb = new double[context.numOutputAttributes()];
            this.postProbOne = new double[context.numOutputAttributes()][kOption.getValue() + 1];
            this.postProbZero = new double[context.numOutputAttributes()][kOption.getValue() + 1];
        } catch(Exception e) {
            System.err.println("Error: no Model Context available.");
            e.printStackTrace();
            System.exit(1);
        }
    }

    @Override
    public void resetLearningImpl() {
        this.window = null;
    }

    @Override
    public void trainOnInstanceImpl(MultiLabelInstance instance) {
        if (this.window == null) {
            this.window = new Instances(instance.dataset());
        }

        if (this.limitOption.getValue() <= this.window.numInstances()) {
            deletePriorCounts(this.window.instance(0));
            this.window.delete(0);
        }

        this.window.add(instance);

        updateRanges(instance);
        updatePriorProb(instance);
        updatePostProb(instance.numberOutputTargets());
    }

    @Override
    public Prediction getPredictionForInstance(MultiLabelInstance instance) {

        MultiLabelPrediction prediction = new MultiLabelPrediction(instance.numberOutputTargets());

        if (this.window != null && this.window.numInstances() > 0) {
            Instances neighbours = kNearestNeighbours(instance, this.window, Math.min(kOption.getValue(), this.window.numInstances()), -1);

            for(int j = 0; j < instance.numberOutputTargets(); j++)
            {
                int count = 0;

                for(int i = 0; i < neighbours.numInstances(); i++) {
                    if(neighbours.instance(i).classValue(j) == 1)
                        count++;
                }

                double probHas = priorProb[j] * postProbOne[j][count];
                double probHasNot = (1.0 - priorProb[j]) * postProbZero[j][count];

                if(probHas > probHasNot)
                    prediction.setVotes(j, new double[] {0.0, 1.0});
                else if (probHas < probHasNot)
                    prediction.setVotes(j, new double[] {1.0, 0.0});
                else
                {
                    int idx = new Random().nextInt(2);
                    prediction.setVotes(j, new double[] {0 == idx ? 1.0 : 0.0, 1 == idx ? 1.0 : 0.0});
                }
            }
        }

        return prediction;
    }

    private void deletePriorCounts(Instance deleted) {
        for(int j = 0; j < deleted.numberOutputTargets(); j++)
            priorCounts[j] = priorCounts[j] - deleted.classValue(j);
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

    private void updatePriorProb(Instance instance) {
        for(int j = 0; j < instance.numberOutputTargets(); j++)
        {
            priorCounts[j] = priorCounts[j] + instance.classValue(j);
            priorProb[j] = (smooth + priorCounts[j]) / (smooth * 2 + this.window.numInstances());
        }
    }

    private void updatePostProb(int numLabels) {

        int k = Math.min(kOption.getValue(), this.window.numInstances() - 1);

        double[][] labelCountOne = new double[numLabels][k + 1];
        double[][] labelCountZero = new double[numLabels][k + 1];

        if(this.window.numInstances() <= 1) {
            for(int l = 0; l < numLabels; l++)
            {
                for(int j = 0; j < k; j++)
                {
                    postProbOne[l][j] = 0.5; //should look into how these should be assigned for the case where there are no neighbors
                    postProbZero[l][j] = 0.5;
                }
            }
        }
        else
        {
            for(int i = 0; i < this.window.numInstances(); i++)
            {
                Instances neighbours = kNearestNeighbours((MultiLabelInstance) this.window.instance(i), this.window, k, i);

                for(int l = 0; l < numLabels; l++)
                {
                    int count = 0;

                    for(int j = 0; j < neighbours.numInstances(); j++) {
                        if(neighbours.instance(j).classValue(l) == 1)
                            count++;
                    }

                    if(this.window.instance(i).classValue(l) == 1)
                        labelCountOne[l][count]++;
                    else
                        labelCountZero[l][count]++;
                }
            }

            for(int l = 0; l < numLabels; l++)
            {
                int sumOne = 0;
                int sumZero = 0;

                for(int j = 0; j <= k; j++)
                {
                    sumOne += labelCountOne[l][j];
                    sumZero += labelCountZero[l][j];
                }

                for(int j = 0; j <= k; j++)
                {
                    postProbOne[l][j] = (smooth + labelCountOne[l][j]) / (smooth * (k + 1) + sumOne);
                    postProbZero[l][j] = (smooth + labelCountZero[l][j]) / (smooth * (k + 1) + sumZero);
                }
            }
        }
    }

    private Instances kNearestNeighbours(MultiLabelInstance instance, Instances window, int k, int index) {
        double[] distances = new double[window.size()];

        for(int i = 0; i < window.size(); i++) {
            if(i == index)
                distances[i] = Double.MAX_VALUE; //cheap way of not counting an instance itself a neighbor
            else
                distances[i] = distance(instance, window.instance(i));
        }

        Instances neighbors = new Instances(window,0,0);

        for(int i = 0; i < k; i++)
        {
            double minDistance = Double.MAX_VALUE;
            int minDistanceInstance = -1;

            for(int j = 0; j < window.size(); j++)
            {
                if(distances[j] < minDistance)
                {
                    minDistance = distances[j];
                    minDistanceInstance = j;
                }
            }

            neighbors.add(window.instance(minDistanceInstance));
            distances[minDistanceInstance] = Double.MAX_VALUE;
        }

        return neighbors;
    }

    private double distance(MultiLabelInstance instance1, Instance instance2) {
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