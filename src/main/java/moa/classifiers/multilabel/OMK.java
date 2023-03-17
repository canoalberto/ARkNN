package moa.classifiers.multilabel;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.*;
import moa.classifiers.AbstractMultiLabelLearner;
import moa.classifiers.MultiLabelClassifier;
import moa.core.Measurement;

import java.util.*;

// https://github.com/cici-xihui/OMK/


public class OMK extends AbstractMultiLabelLearner implements MultiLabelClassifier {

    public IntOption windowSizeValue = new IntOption("WindowSize", 'w', "The maximum number of recent instances to store", 100);
    public IntOption initialOption = new IntOption("SI", 'i', "number of instances for initializing", 1000);
    public IntOption kOption = new IntOption("k", 'k', "Size of clusters", 30);
    public IntOption rsOption = new IntOption("ReservoirSize", 'r', "Size of reservoir sampling for each clusters",  100);
    public IntOption nnOption = new IntOption("nn", 'n', "The n nearest neighbors for prediction",  3);


    private int windowSize;
    private int num_initial;
    public int size_kernels;
    public int size_RS;
    public int size_nn;
    private List<Instance> window;
    private List<Instance> initialStream;
    private ArrayList<Instance> kernels;
    private ArrayList<ArrayList<Instance>> reservoirsamp_Mat;
    private ArrayList<Integer> countsInsertList;

    private Random ran = new Random(2);

    public OMK() {
    }

    @Override
    public void resetLearningImpl() {

        windowSize = windowSizeValue.getValue();
        num_initial = initialOption.getValue();

        size_RS = rsOption.getValue();
        size_kernels = kOption.getValue();
        size_nn = nnOption.getValue();

        window = new ArrayList<Instance>();
        initialStream = new ArrayList<Instance>();
        kernels = new ArrayList<Instance>();
        reservoirsamp_Mat = new ArrayList<ArrayList<Instance>>();
        countsInsertList = new ArrayList<Integer>();


    }


    /**
     * @param centreIndex The list of index of intial example in initialStream
     */
    public void initializeImpl(ArrayList<Integer> centreIndex) {

        int i;
        for(i = 0; i < centreIndex.size(); i++){

            kernels.add(initialStream.get(centreIndex.get(i)));

            ArrayList<Instance> rsList = new ArrayList<Instance>();
            rsList.add(initialStream.get(centreIndex.get(i)));

            reservoirsamp_Mat.add(rsList);
            countsInsertList.add(1);
        }

        for(i = 0; i < num_initial; i++ ){
            if(! centreIndex.contains(i)){
                Instance newInstance = initialStream.get(i);
                updateCluster(newInstance);
            }
        }
    }

    @Override
    public void trainOnInstanceImpl(MultiLabelInstance multiLabelInstance) {

        Instance instance = sparseTodense(multiLabelInstance);
        if(initialStream.size() < num_initial ) {
            initialStream.add(instance);
            // Initialization
        }else if (initialStream.size() == num_initial){
            ArrayList<Integer> centreIndex = chooseRandomCentres(initialStream);
            initializeImpl(centreIndex);
            num_initial = -1;
        }

        if (num_initial == -1){
            updateCluster(instance);
        }

        if(window.size() == windowSize){
            // Update Windows
            window.remove(0);
        }
        window.add(instance);

    }


    private static double getdistance(double[] pointA, double[] pointB) {

        double distance = 0.0D;

        for(int i = 0; i < pointA.length; ++i) {
            double d = pointA[i] - pointB[i];
            distance += d * d;
        }

        return Math.sqrt(distance);
    }

    private double[] instanceAttributes(Instance inst){

        double[] inst_attributes = new double[inst.numInputAttributes()];

        for(int i = 0; i < inst.numInputAttributes(); i++) {
            inst_attributes[i] = inst.valueInputAttribute(i);
        }

        return inst_attributes;
    }

    private double[] instanceLabels(Instance inst){

        double[] inst_labels = new double[inst.numberOutputTargets()];

        for(int i = 0; i < inst.numberOutputTargets(); i++) {
            inst_labels[i] = inst.classValue(i);
        }

        return inst_labels;
    }

    protected Instance sparseTodense(Instance inst) {

        if(inst.numValues() == inst.numAttributes()){
            return inst;
        } else {
            InstancesHeader header = new InstancesHeader(inst.dataset());
            double[] attVals = new double[inst.numAttributes()];

            for (int attributeIndex = 0; attributeIndex < inst.numAttributes(); ++attributeIndex) {
                attVals[attributeIndex] = inst.value(attributeIndex);
            }
            Instance instance = new DenseInstance(1.0D, attVals);
            instance.setDataset(header);

            return instance;
        }
    }

    private ArrayList<Integer> chooseRandomCentres(List<Instance> initialStream) {

        int k = size_kernels;

        ArrayList<Integer> centres = new ArrayList<Integer>();
        int i = 0;
        centres.add(i);
        double[] newcentre = instanceLabels(initialStream.get(i));

        int streamL = initialStream.size();
        double[] Streamdist = new double[streamL];

        for(i = 0; i < streamL; ++i) {
            Streamdist[i] = getdistance(instanceLabels(initialStream.get(i)), newcentre);
        }

        for(i = 1; i < k; ++i) {
            double cost = 0.0D;

            int j;
            for(j = 0; j < streamL; ++j) {
                cost += Streamdist[j];
            }

            double sum = 0.0D;
            int pos;
            if(cost > 0.0D) {
                do {
                    double random = ran.nextDouble();
                    sum = 0.0D;
                    pos = -1;

                    for (j = 0; j < streamL; ++j) {
                        sum += Streamdist[j];
                        if (random <= sum / cost) {
                            pos = j;
                            break;
                        }
                    }
                } while (pos < 0);
            }else{
                do {
                    pos = ran.nextInt(streamL);

                }while (centres.contains(pos));
            }
            centres.add(pos);
            newcentre = instanceLabels(initialStream.get(pos));

            for(j = 0; j < streamL; ++j) {
                double newdist = getdistance(instanceLabels(initialStream.get(j)), newcentre);
                if (Streamdist[j] > newdist) {
                    Streamdist[j] = newdist;
                }
            }
        }
        return centres;
    }


    /**
     * Returns the n smallest indices of the smallest values (sorted).
     */
    private int[] nArgMin(int n, double[] distances) {

        int indices[] = new int[n];

        for (int i = 0; i < n; i++){
            double minValue = Double.MAX_VALUE;
            for (int j = 0; j < distances.length; j++){

                if (distances[j] < minValue){
                    boolean alreadyUsed = false;
                    for (int k = 0; k < i; k++){
                        if (indices[k] == j){
                            alreadyUsed = true;
                        }
                    }
                    if (!alreadyUsed){
                        indices[i] = j;
                        minValue = distances[j];
                    }
                }
            }
        }
        return indices;
    }

    /**
     * Computes the Euclidean distance between one sample and a collection of samples in an 1D-array.
     */
    private double[] get1ToNDistances(List<Instance> kernelsList, Instance multiLabelInstance, char model ) {

        List<double[]> pointAList = new ArrayList<double[]>();
        double[] pointA;
        double[] pointB;
        double[] distances = new double[kernelsList.size()];

        if(model == 'U'){
            for (int i = 0; i < kernelsList.size(); ++i) {
                pointA = instanceLabels(kernelsList.get(i));
                pointAList.add(pointA);
            }
            pointB = instanceLabels(multiLabelInstance);
        }
        else{
            for (int i = 0; i < kernelsList.size(); ++i) {
                pointA = instanceAttributes(kernelsList.get(i));
                pointAList.add(pointA);
            }
            pointB = instanceAttributes(multiLabelInstance);

        }

        for (int i = 0; i < pointAList.size(); i++) {
            pointA = pointAList.get(i);
            distances[i] = getdistance(pointA, pointB);
        }
        return distances;
    }


    /**
     *
     * @param kernelsIndex
     * Traverse all the example in the reservoir sampling and get the average for features and labels
     */
    private void calculateCenter(int kernelsIndex, Instance multiLabelInstance) {

        Instance kernel = kernels.get(kernelsIndex);

        int numAtt = kernel.numAttributes();
        int numMemory = reservoirsamp_Mat.get(kernelsIndex).size();
        double[] res = new double[numAtt];

        if (numMemory == size_RS){
            numMemory --;
        }

        for (int i = 0; i < numAtt; ++i) {
            res[i] = kernels.get(kernelsIndex).value(i) * (double) numMemory;
            res[i] += multiLabelInstance.value(i);
            res[i] /= (double) (numMemory + 1);
            kernel.setValue(i, res[i]);
        }

        kernels.set(kernelsIndex, kernel);

    }

    private void updateCluster(Instance multiLabelInstance) {

        char model = 'U';
        double[] distances = get1ToNDistances(kernels, multiLabelInstance, model);
        int kernelsIndex = nArgMin(1, distances)[0];
        int num_insert = countsInsertList.get(kernelsIndex);

        int replace;
        if (num_insert < size_RS) {
            reservoirsamp_Mat.get(kernelsIndex).add(multiLabelInstance);

        }else{

            replace = ran.nextInt(num_insert);

            if (replace < size_RS){
                reservoirsamp_Mat.get(kernelsIndex).set(replace, multiLabelInstance);
            }
        }

        calculateCenter(kernelsIndex, multiLabelInstance);
        countsInsertList.set(kernelsIndex,countsInsertList.get(kernelsIndex)+1);
    }


    @Override
    public Prediction getPredictionForInstance(MultiLabelInstance multiLabelInstance) {

        Instance instance = sparseTodense(multiLabelInstance);

        int numLabels = multiLabelInstance.numberOutputTargets();
        MultiLabelPrediction prediction = new MultiLabelPrediction(numLabels);

        char model = 'P';
        List<Instance> preInstances = new ArrayList<Instance>();
        if (kernels.size() != 0 ) {
            double[] distances = get1ToNDistances(kernels, multiLabelInstance, model);
            int kernelsIndex = nArgMin(1, distances)[0];
            for(int i = 0; i < reservoirsamp_Mat.get(kernelsIndex).size(); i++)
                preInstances.add(reservoirsamp_Mat.get(kernelsIndex).get(i));
            for(int j = 0; j < window.size(); j++)
                preInstances.add(window.get(j));

        }else{
            for(int j = 0; j < initialStream.size(); j++)
                preInstances.add(initialStream.get(j));
        }


        double[] preDistance = get1ToNDistances(preInstances, multiLabelInstance, model);

        int[] nnIndices = nArgMin(Math.min(preDistance.length, size_nn), preDistance);
        for(int j = 0; j < numLabels; j++)
        {
            int count = 0;

            for (int nnIdx : nnIndices){
                if (preInstances.get(nnIdx).classValue(j) == 1)
                    count++;
            }

            double relativeFrequency = count / (double) (size_nn);

            prediction.setVotes(j, new double[]{1.0 - relativeFrequency, relativeFrequency});
        }

        return prediction;

    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[0];
    }

    @Override
    public void getModelDescription(StringBuilder stringBuilder, int i) {

    }

    @Override
    public boolean isRandomizable() {
        return true;
    }

}
