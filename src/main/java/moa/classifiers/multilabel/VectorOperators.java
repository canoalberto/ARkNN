package moa.classifiers.multilabel;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import com.yahoo.labs.samoa.instances.SparseInstance;
import moa.core.DoubleVector;

import java.util.ArrayList;


public class VectorOperators {


    public static double dotProd(Instance inst1, DoubleVector weights) {
        double result = 0.0D;
        int n1 = inst1.numValues();
        int n2 = weights.numValues();
        int p1 = 0;
        int p2 = 0;

        while(p1 < n1 && p2 < n2) {
            int ind1 = inst1.index(p1);
            if (inst1.classIndex() == 0) {
                if (ind1 < inst1.numOutputAttributes()) {
                    p1++;
                    continue;
                }else{
                    if ((ind1-inst1.classIndex()) == p2) {
                        result += inst1.valueSparse(p1) * weights.getValue(p2);
                        ++p1;
                        ++p2;
                    } else if ((ind1-inst1.numOutputAttributes()) > p2) {
                        ++p2;
                    } else {
                        ++p1;
                    }
                }
            }else if (ind1 < inst1.classIndex()) {
                if (ind1 == p2) {
                    result += inst1.valueSparse(p1) * weights.getValue(p2);
                    ++p1;
                    ++p2;
                } else if (ind1 > p2) {
                    ++p2;
                } else {
                    ++p1;
                }
            }
        }

        return result;
    }

    /**
     * Cosine Similarity  of feature vectors
     */
    public static double getCosAtt(Instance instance1, Instance instance2, double[] attributeRangeMax, double[] attributeRangeMin ) {

        double distance = 0.0D;
        double distanceA = 0.0D;
        double distanceB = 0.0D;

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
                    distance += val1 * val2;
                    distanceA += val1 * val1;
                    distanceB += val2 * val2;
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

                int idx1, idx2;

                if (instance1.classIndex() == 0) {
                    idx1 = firstI - numOutputs;
                    idx2 = secondI - numOutputs;
                }else{
                    idx1 = firstI;
                    idx2 = secondI;
                }

                if (instance1.classIndex() == 0) {
                    if (firstI < numOutputs) {
                        p1++;
                        continue;
                    }
                    if (secondI < numOutputs) {
                        p2++;
                        continue;
                    }
                }else{
                    if (firstI >= instance1.classIndex() && secondI >= instance1.classIndex()) {
                        break;
                    }
                }

                if(firstI == secondI) {


                    if(attributeRangeMax[idx1] - attributeRangeMin[idx1] != 0)
                    {
                        double val1 = instance1.valueSparse(p1);
                        double val2 = instance2.valueSparse(p2);
                        val1 = (val1 - attributeRangeMin[idx1]) / (attributeRangeMax[idx1] - attributeRangeMin[idx1]);
                        val2 = (val2 - attributeRangeMin[idx1]) / (attributeRangeMax[idx1] - attributeRangeMin[idx1]);
                        distance += val1 * val2;
                        distanceA += val1 * val1;
                        distanceB += val2 * val2;
                    }
                    p1++;
                    p2++;

                } else if (firstI > secondI) {
                    //System.out.println(instance1.classIndex());

                    if(attributeRangeMax[idx2] - attributeRangeMin[idx2] != 0)
                    {
                        double val2 = instance2.valueSparse(p2);
                        val2 = (val2 - attributeRangeMin[idx2]) / (attributeRangeMax[idx2] - attributeRangeMin[idx2]);
                        distanceB += val2 * val2;
                    }
                    p2++;
                } else {

                    if(attributeRangeMax[idx1] - attributeRangeMin[idx1] != 0)
                    {
                        double val1 = instance1.valueSparse(p1);
                        val1 = (val1 - attributeRangeMin[idx1]) / (attributeRangeMax[idx1] - attributeRangeMin[idx1]);
                        distanceA += val1 * val1;
                    }
                    p1++;

                }
            }
        }

        distance = distance / (Math.sqrt(distanceA) * Math.sqrt(distanceB));

        return 1 - distance;
    }


    /**
     * Cosine similarity of feature vectors
     */
    public static double getCosLab(Instance instance1, Instance instance2) {

        double distance = 0.0D;
        double distanceA = 0.0D;
        double distanceB = 0.0D;


        if(instance1.numValues() == instance1.numAttributes()) // Dense Instance
        {
            for(int i = 0; i < instance1.numberOutputTargets(); i++)  // erreur, if instance1.numberOutputTargets() != instance2.numberOutputTargets()
            {
                double val1 = instance1.classValue(i);
                double val2 = instance2.classValue(i);
                distance += val1 * val2;
                distanceA += val1 * val1;
                distanceB += val2 * val2;

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

                if (instance1.classIndex() == 0) {
                    if (firstI >= numOutputs && secondI >= numOutputs) {
                        break;
                    }
                }else {
                    if (firstI < instance1.classIndex()) {
                        p1++;
                        continue;
                    }
                    if (secondI < instance1.classIndex()) {
                        p2++;
                        continue;
                    }
                }

                if(firstI == secondI) {

                    double val1 = instance1.valueSparse(p1);
                    double val2 = instance2.valueSparse(p2);

                    distance += val1 * val2;
                    distanceA += val1 * val1;
                    distanceB += val2 * val2;

                    p1++;
                    p2++;
                } else if (firstI > secondI) {
                    double val2 = instance2.valueSparse(p2);
                    distanceB += val2 * val2;
                    p2++;
                } else {
                    double val1 = instance1.valueSparse(p1);
                    distanceA += val1 * val1;
                    p1++;
                }
            }

        }

        distance = distance / (Math.sqrt(distanceA) * Math.sqrt(distanceB));

        return 1 - distance;

    }

    /**
     * Distance Euclidienne of feature vectors
     */
    public static double getDistanceAtt(Instance instance1, Instance instance2, double[] attributeRangeMax, double[] attributeRangeMin) {

        double distance = 0.0D;

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
                    distance += (val1 - val2)*(val1 - val2);
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

                if (instance1.classIndex() == 0) {
                    if (firstI < numOutputs) {
                        p1++;
                        continue;
                    }
                    if (secondI < numOutputs) {
                        p2++;
                        continue;
                    }
                }else{
                    if (firstI > instance1.classIndex() && secondI > instance1.classIndex()) {
                        break;
                    }
                }

                int idx;
                if(firstI == secondI) {
                    if (instance1.classIndex() == 0) {
                        idx = firstI - numOutputs;
                    }else{
                        idx = firstI;
                    }

                    if(attributeRangeMax[idx] - attributeRangeMin[idx] != 0)
                    {
                        double val1 = instance1.valueSparse(p1);
                        double val2 = instance2.valueSparse(p2);
                        val1 = (val1 - attributeRangeMin[idx]) / (attributeRangeMax[idx] - attributeRangeMin[idx]);
                        val2 = (val2 - attributeRangeMin[idx]) / (attributeRangeMax[idx] - attributeRangeMin[idx]);
                        distance += (val1 - val2)*(val1 - val2);
                    }
                    p1++;
                    p2++;

                } else if (firstI > secondI) {
                    if (instance1.classIndex() == 0) {
                        idx = secondI - numOutputs;
                    }else{
                        idx = secondI;
                    }

                    if(attributeRangeMax[idx] - attributeRangeMin[idx] != 0)
                    {
                        double val2 = instance2.valueSparse(p2);
                        val2 = (val2 - attributeRangeMin[idx]) / (attributeRangeMax[idx] - attributeRangeMin[idx]);
                        distance += val2 * val2;
                    }
                    p2++;
                } else {
                    if (instance1.classIndex() == 0) {
                        idx = firstI - numOutputs;
                    }else{
                        idx = firstI;
                    }
                    if(attributeRangeMax[idx] - attributeRangeMin[idx] != 0)
                    {
                        double val1 = instance1.valueSparse(p1);
                        val1 = (val1 - attributeRangeMin[idx]) / (attributeRangeMax[idx] - attributeRangeMin[idx]);
                        distance += val1 * val1;
                    }
                    p1++;

                }
            }
        }

        return distance;
    }


    /**
     * Distance Euclidienne of feature vectors
     */
    public static double getDistanceLab(Instance instance1, Instance instance2) {

        double distance = 0.0D;

        if(instance1.numValues() == instance1.numAttributes()) // Dense Instance
        {
            for(int i = 0; i < instance1.numberOutputTargets(); i++)  // erreur, if instance1.numberOutputTargets() != instance2.numberOutputTargets()
            {
                double val1 = instance1.classValue(i);
                double val2 = instance2.classValue(i);
                distance += (val1 - val2)*(val1 - val2);

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

                if (instance1.classIndex() == 0) {
                    if (firstI >= numOutputs && secondI >= numOutputs) {
                        break;
                    }
                }else {
                    if (firstI < instance1.classIndex()) {
                        p1++;
                        continue;
                    }
                    if (secondI < instance1.classIndex()) {
                        p2++;
                        continue;
                    }
                }

                if(firstI == secondI) {
                    double val1 = instance1.valueSparse(p1);
                    double val2 = instance2.valueSparse(p2);
                    distance += (val1 - val2)*(val1 - val2);
                    p1++;
                    p2++;

                } else if (firstI > secondI) {
                    double val2 = instance2.valueSparse(p2);
                    distance += val2 * val2;
                    p2++;
                } else {
                    double val1 = instance1.valueSparse(p1);
                    distance += val1 * val1;
                    p1++;
                }
            }

        }

        return distance;

    }


    public static void printKernels(ArrayList<Instance> kernels) {
        if(kernels.size() == 0){
            System.out.println(" null; \n");
        }else {
            for (int i = 0; i < kernels.size(); i++) {
            }
        }
    }


    public static Instance calculateCenter(Instance kernel, Instance multiLabelInstance, int numMemory, boolean learning) {

        int rate = 1;
        if(!learning) rate = -1;

        if(kernel.numValues() == kernel.numAttributes()) // Dense Instance
        {
            int numAtt = kernel.numAttributes();
            double[] res = new double[numAtt];
            for (int i = 0; i < numAtt; ++i) {
                res[i] = kernel.value(i) * (double) numMemory;
                res[i] += rate*multiLabelInstance.value(i);
                res[i] /= (double) (numMemory + 1);
                kernel.setValue(i, res[i]);
            }
            return kernel;

        }
        else // Sparse Instance
        {
            ArrayList<Integer> indexValues = new ArrayList<Integer>();
            ArrayList<Double> attributeValues = new ArrayList<Double>();

            int firstI = -1, secondI = -1;
            int firstNumValues = kernel.numValues();
            int secondNumValues = multiLabelInstance.numValues();
            int numAttributes = kernel.numAttributes();

            for (int p1 = 0, p2 = 0; p1 < firstNumValues || p2 < secondNumValues;) {

                if (p1 >= firstNumValues) {
                    firstI = numAttributes;
                } else {
                    firstI = kernel.index(p1);
                }

                if (p2 >= secondNumValues) {
                    secondI = numAttributes;
                } else {
                    secondI = multiLabelInstance.index(p2);
                }

                if (firstI == secondI) {
                    double centroid = kernel.valueSparse(p1) * (double) numMemory;
                    centroid += rate*multiLabelInstance.valueSparse(p2);
                    centroid /= (double) (numMemory + 1);
                    attributeValues.add(centroid);
                    indexValues.add(firstI);
                    p1++;
                    p2++;
                } else if (firstI > secondI) {

                    double centroid = rate * multiLabelInstance.valueSparse(p2);
                    centroid /= (double) (numMemory + 1);
                    attributeValues.add(centroid);
                    indexValues.add(secondI);

                    p2++;
                } else {
                    double centroid = kernel.valueSparse(p1) * (double) numMemory;;
                    centroid /= (double) (numMemory + 1);
                    attributeValues.add(centroid);
                    indexValues.add(firstI);
                    p1++;
                }
            }

            int[] index = new int[attributeValues.size()];
            double[] value = new double[attributeValues.size()];

            for (int i = 0; i < attributeValues.size(); i++) {
                index[i] = indexValues.get(i);
                value[i] = attributeValues.get(i);
            }

            Instance newkernel = new SparseInstance(1.0, value, index, numAttributes);
            InstancesHeader header = new InstancesHeader(multiLabelInstance.dataset());
            newkernel.setDataset(header);
            return newkernel;


        }
    }

    /**
     * Returns the kernel with n biggest values (sorted).
     */
    public static Instance kernelMax(Instance kernel) {
        Instance newkernel = kernel;
        //int indices[] = new int[n];
        if(kernel.numValues() != kernel.numAttributes()){
            int firstI = -1;
            int NumValues  = kernel.numValues();
            int numAttributes   = kernel.numAttributes();
            int numOutputs      = kernel.numOutputAttributes();
            double maxValue = 0.01;

            for (int p1 = 0; p1 < NumValues; p1++) {

                if (p1 >= NumValues) {
                    firstI = numAttributes;
                } else {
                    firstI = kernel.index(p1);
                }

                if (kernel.classIndex() == 0) {
                    if (firstI < numOutputs) {
                        p1++;
                        continue;
                    }
                }else{
                    if (firstI >= kernel.classIndex()) {
                        break;
                    }
                }

                if (kernel.valueSparse(p1) < maxValue) {
                    newkernel.setValue(firstI,0);
                    }
                }
        }


        return newkernel;
    }
}