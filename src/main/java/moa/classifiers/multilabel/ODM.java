package moa.classifiers.multilabel;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.*;
import moa.classifiers.AbstractMultiLabelLearner;
import moa.classifiers.MultiLabelClassifier;
import moa.core.Measurement;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

// https://github.com/cici-xihui/ODM/

public class ODM extends AbstractMultiLabelLearner implements MultiLabelClassifier  {
	
    public IntOption RandomSeedOption = new IntOption("RandomSeed", 's', "Seed for random.", 2);
    public IntOption windowSizeValue = new IntOption("WindowSize", 'w', "The maximum number of recent instances to store", 50);
    public IntOption initialOption = new IntOption("IS", 'i', "number of instances for initializing", 1000);
    public IntOption kOption = new IntOption("k", 'k', "Size of clusters", 40);
    public IntOption rsOption = new IntOption("ReservoirSize", 'r', "Size of reservoir sampling for each clusters",  100);
    public IntOption nnOption = new IntOption("nn", 'n', "The n nearest neighbors for prediction",  3);

    private Random ran;
    private int numLabels;
    private int windowSize;
    private int num_initial;
    private int size_kernels;
    private int size_RS;
    private int size_nn;

    private List<Instance> window;
    private List<Instance> initialStream;
    private ArrayList<Instance> kernels;
    private ArrayList<ArrayList<Instance>> reservoirs_Mat;
    private ArrayList<Integer> countsInsertList;
    private ArrayList<Double> weight_ML;
    private ArrayList<Double> weight_MS;

    MultiLabelPrediction short_predict;
    MultiLabelPrediction long_predict;
    private int predictIndex;

    private double[] attributeRangeMin;
    private double[] attributeRangeMax;


    public ODM() {
    }

    @Override
    public void setModelContext(InstancesHeader context) {
        try {
            this.numLabels = context.numOutputAttributes();
            this.short_predict = new MultiLabelPrediction(this.numLabels);
            this.long_predict = new MultiLabelPrediction(this.numLabels);

            this.attributeRangeMin = new double[context.numInputAttributes()];
            this.attributeRangeMax = new double[context.numInputAttributes()];

        } catch(Exception e) {
            System.err.println("Error: no Model Context available.");
            e.printStackTrace();
            System.exit(1);
        }
    }

    @Override
    public void resetLearningImpl() {

        this.ran = new Random(RandomSeedOption.getValue());
        this.windowSize = this.windowSizeValue.getValue();
        this.num_initial = this.initialOption.getValue();

        this.size_RS = this.rsOption.getValue();
        this.size_kernels = this.kOption.getValue();
        this.size_nn = this.nnOption.getValue();

        this.window = new ArrayList<Instance>();
        this.initialStream = new ArrayList<Instance>();
        this.kernels = new ArrayList<Instance>();
        this.reservoirs_Mat = new ArrayList<ArrayList<Instance>>();
        this.countsInsertList = new ArrayList<Integer>();
        this.weight_ML = new ArrayList<Double>();
        this.weight_MS = new ArrayList<Double>();
    }


    /**
     * @param centreIndex The list of index of intial example in initialStream
     */
    public void initializeImpl(ArrayList<Integer> centreIndex) {

        int i;
        for(i = 0; i < centreIndex.size(); i++){
            this.kernels.add(this.initialStream.get(centreIndex.get(i)));

            ArrayList<Instance> rsList = new ArrayList<Instance>();
            rsList.add(this.initialStream.get(centreIndex.get(i)));
            this.reservoirs_Mat.add(rsList);
            this.countsInsertList.add(1);
        }
        for(i = 0; i < this.num_initial; i++ ){
            if(! centreIndex.contains(i)){
                Instance newInstance = this.initialStream.get(i);
                updateCluster(newInstance);
            }
        }
    }

    @Override
    public void trainOnInstanceImpl(MultiLabelInstance instance) {

        updateRanges(instance);

        if (this.initialStream != null){
            if(this.initialStream.size() < this.num_initial ) {
            this.initialStream.add(instance);
            }else if (this.initialStream.size() == this.num_initial) {
                ArrayList<Integer> centreIndex = chooseRandomCentres(this.initialStream);
                initializeImpl(centreIndex);
                this.initialStream = null;
            }
        }else {
            updateHistories(instance);
            updateCluster(instance);
        }


        if(this.window.size() == this.windowSize){
            this.window.remove(0);
        }
        this.window.add(instance);

    }



    private ArrayList<Integer> chooseRandomCentres(List<Instance> initialStream) {

        ArrayList<Integer> centres = new ArrayList<Integer>();
        int i = 0;
        centres.add(i);
        Instance newcentre = initialStream.get(i);

        int streamL = initialStream.size();
        double[] Streamdist = new double[streamL];

        for(i = 0; i < streamL; ++i){
            Streamdist[i] = VectorOperators.getCosLab(initialStream.get(i), newcentre);
            //Streamdist[i] = VectorOperators.getDistanceLab(initialStream.get(i), newcentre);
        }

        for(i = 1; i < this.size_kernels; ++i) {
            double cost = 0.0D;

            int j;
            for(j = 0; j < streamL; ++j) {
                cost += Streamdist[j];
            }

            double sum = 0.0D;
            int pos;
            if(cost > 0.0D) {
                do {
                    double random = this.ran.nextDouble();
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
                    pos = this.ran.nextInt(streamL);

                }while (centres.contains(pos));
            }
            centres.add(pos);
            newcentre = initialStream.get(pos);

            for(j = 0; j < streamL; ++j) {
                double newdist = VectorOperators.getCosLab(initialStream.get(j), newcentre);
                //double newdist = VectorOperators.getDistanceLab(initialStream.get(j), newcentre);
                if (Streamdist[j] > newdist) {
                    Streamdist[j] = newdist;
                }
            }
        }
        return centres;
    }

    private void updateRanges(MultiLabelInstance instance) {
        for(int i = 0; i < instance.numInputAttributes(); i++) {
            if(instance.valueInputAttribute(i) < this.attributeRangeMin[i])
                this.attributeRangeMin[i] = instance.valueInputAttribute(i);
            if(instance.valueInputAttribute(i) > this.attributeRangeMax[i])
                this.attributeRangeMax[i] = instance.valueInputAttribute(i);
        }
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
     * Computes the distance between one sample and a collection of samples in an 1D-array.
     */

    private double[] get1ToNDistances(List<Instance> kernelsList, Instance multiLabelInstance, char model ) {

        double[] distances = new double[kernelsList.size()];

        if(model == 'U'){
            for (int i = 0; i < kernelsList.size(); ++i) {
                distances[i] = VectorOperators.getCosLab(kernelsList.get(i),multiLabelInstance);
                //distances[i] = VectorOperators.getDistanceLab(kernelsList.get(i),multiLabelInstance);
            }
        }
        else{
            for (int i = 0; i < kernelsList.size(); ++i) {
                distances[i] = VectorOperators.getCosAtt(kernelsList.get(i),multiLabelInstance, this.attributeRangeMax, this.attributeRangeMin);
                //distances[i] = VectorOperators.getDistanceAtt(kernelsList.get(i),multiLabelInstance, attributeRangeMax, attributeRangeMin);
            }

        }

        return distances;
    }

    /**
     *
     * @param kernelsIndex
     * Traverse all the example in the reservoir sampling and get the average for features and labels
     */

    private void updateCenter(int kernelsIndex, Instance newInstance, boolean learning) {

        int numMemory = this.countsInsertList.get(kernelsIndex);
        Instance newkernel = VectorOperators.calculateCenter(this.kernels.get(kernelsIndex), newInstance, numMemory, learning);

        this.kernels.set(kernelsIndex, newkernel);

    }


    private void updateCluster(Instance multiLabelInstance) {

        double[] updateDistances = get1ToNDistances(this.kernels, multiLabelInstance, 'U');
        int updateIndex = nArgMin(1, updateDistances)[0];

        int num_insert = this.countsInsertList.get(updateIndex);

        if (num_insert < this.size_RS) {
            this.reservoirs_Mat.get(updateIndex).add(multiLabelInstance);
        } else{

            int replace = this.ran.nextInt(num_insert);
            if (replace < this.size_RS){
                this.reservoirs_Mat.get(updateIndex).set(replace, multiLabelInstance);
            }
        }

        updateCenter(updateIndex, multiLabelInstance, true);
        this.countsInsertList.set(updateIndex,num_insert+1);

        if(updateIndex != this.predictIndex){
            updateCenter(this.predictIndex, multiLabelInstance, false);
        }

    }


    @Override
    public Prediction getPredictionForInstance(MultiLabelInstance multiLabelInstance) {

        MultiLabelPrediction prediction = new MultiLabelPrediction(this.numLabels);

        double weightST = 1.0D;
        double weightLT = 1.0D;

        if (this.kernels.size() != 0 && this.weight_ML.size() > 0) {

            for (int i = 0; i < this.weight_ML.size(); i++) {
                weightST += this.weight_MS.get(i);
                weightLT += this.weight_ML.get(i);
            }

            double sumWeight = weightST + weightLT;
            if(sumWeight != 0){
                weightST = weightST/sumWeight;
                weightLT = weightLT/sumWeight;
            }else {
                weightST = 1.0D;
                weightLT = 1.0D;
            }

            double[] distances = get1ToNDistances(this.kernels, multiLabelInstance, 'P');
            this.predictIndex = nArgMin(1, distances)[0];
            this.long_predict = getPrediction(multiLabelInstance, this.reservoirs_Mat.get(this.predictIndex));
            this.short_predict = getPrediction(multiLabelInstance, this.window);

            for(int j = 0; j < this.numLabels; j++) {
                double count = 0;
                count += weightLT * this.long_predict.getVote(j,1) + weightST* this.short_predict.getVote(j,1);
                prediction.setVotes(j, new double[]{1.0 - count, count});
            }

        }else {
            prediction = getPrediction(multiLabelInstance, this.window);
            this.short_predict= prediction;
        }
        return prediction;

    }

    protected MultiLabelPrediction getPrediction(MultiLabelInstance multiLabelInstance, List<Instance> preInstances){

        MultiLabelPrediction prediction = new MultiLabelPrediction(this.numLabels);
        double[] preDistance = get1ToNDistances(preInstances, multiLabelInstance, 'P');

        int[] nnIndices = nArgMin(Math.min(preDistance.length, this.size_nn), preDistance);
        for(int j = 0; j < this.numLabels; j++) {

            int count = 0;
            for (int nnIdx : nnIndices){
                if (preInstances.get(nnIdx).classValue(j) == 1)
                    count++;
            }
            double relativeFrequency = count / (double) (this.size_nn);
            prediction.setVotes(j, new double[]{1.0 - relativeFrequency, relativeFrequency});
        }
        return prediction;
    }

    protected void updateHistories(MultiLabelInstance multiLabelInstance) {

        double shortAcc = AccPred(this.short_predict, multiLabelInstance);
        if (this.weight_MS.size() == this.windowSize) {
            this.weight_MS.remove(0);
        }
        this.weight_MS.add(shortAcc);

        double longAcc = AccPred(this.long_predict, multiLabelInstance);
        if (this.weight_ML.size() == this.windowSize) {
            this.weight_ML.remove(0);
        }
        this.weight_ML.add(longAcc);
    }

    public static double AccPred(MultiLabelPrediction pred, Instance instance) {

        double cur_tp = 0;
        double cur_fp = 0;
        double cur_fn = 0;
        double delta_exAcc = 0.0;
        for(int i = 0; i < pred.size(); i++) {
            double yp = (pred.getVote(i,1) >= 0.5) ? 1 : 0; ;
            double yt = instance.classValue(i);
            cur_tp   += (yt == 1 && yp == 1) ? 1 : 0;
            cur_fn   += (yt == 1 && yp == 0) ? 1 : 0;
            cur_fp   += (yt == 0 && yp == 1) ? 1 : 0;

        }
        delta_exAcc = cur_tp / (cur_tp + cur_fn + cur_fp);

        return delta_exAcc;

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