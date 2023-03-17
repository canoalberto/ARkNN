package experiments;

import utils.Utils;

public class Experiments {

	public static void main(String[] args) throws Exception {

		String[] datasets = new String[] {
				"Birds","Emotions","VirusGO","Flags","Scene",
				"Enron","Genbase","Medical","tmc2007-500","Water-quality","Yeast",
				"20NG","Corel5k","EukaryotePseAAC","PlantPseAAC","Reuters-K500","Mediamill",
				"Ohsumed","CAL500","Yelp","Slashdot","HumanPseAAC","Langlog","GnegativePseAAC","CHD_49","Stackex_chess",
				"Corel16k001","Bibtex","Imdb","Nuswide_cVLADplus","Nuswide_BoW","Yahoo_Society","Eurlex-sm","Yahoo-Computers","Bookmarks",
				"Hypersphere","Hypercube",
		};

		int[] numberLabels = new int[] {
				19,6,6,7,6,
				53,27,45,22,14,14,
				20,374,22,12,103,101,
				23,174,5,22,14,75,8,6,227,
				153,159,28,81,81,27,201,33,208,
				10,10
		};
		
		String[] algorithms = new String[] {
				"moa.classifiers.multilabel.MLkNN",
				"moa.classifiers.multilabel.MLSAMkNN",
				"moa.classifiers.multilabel.MLSAMPkNN",
				"moa.classifiers.multilabel.MLSAkNN",
				"moa.classifiers.multilabel.meta.AESAKNNS",
				"moa.classifiers.multilabel.ODM",
				"moa.classifiers.multilabel.OMK",
				"moa.classifiers.multilabel.ARkNN",
		};

		String[] algorithmNames = new String[] {
				"MLkNN",
				"MLSAMkNN",
				"MLSAMPkNN",
				"MLSAkNN",
				"AESAKNNS",
				"ODM",
				"OMK",
				"ARkNN",
		};
		
		// Executables
		System.out.println("===== Executables =====");
		for(int dat = 0; dat < datasets.length; dat++) {
			for(int alg = 0; alg < algorithmNames.length; alg++)
			{
				String memory = "-XX:ParallelGCThreads=12 -Xms16g -Xmx128g";

				System.out.println("java " + memory + " -javaagent:sizeofag-1.0.4.jar -cp ARkNN-1.0-jar-with-dependencies.jar "
						+ "moa.DoTask EvaluatePrequentialMultiLabel "
						+ " -e \"(PrequentialMultiLabelPerformanceEvaluator)\""
						+ " -s \"(MultiTargetArffFileStream -c " + numberLabels[dat] + " -f datasets/" + datasets[dat] + ".arff)\"" 
						+ " -l \"(" + algorithms[alg] + ")\""
						+ " -f 100"
						+ " -d results/" + algorithmNames[alg] + "-" + datasets[dat] + ".csv");
			}
		}
		
		// Show metrics for results
		System.out.println("===== Results =====");
		
		Utils.metric("Subset Accuracy", "averaged", "results", algorithmNames, datasets);
		Utils.metric("Example-Based Accuracy", "averaged", "results", algorithmNames, datasets);
		Utils.metric("Example-Based F-Measure", "averaged", "results", algorithmNames, datasets);
		
		Utils.metric("evaluation time (cpu seconds)", "last", "results", algorithmNames, datasets);
		Utils.metric("model cost (RAM-Hours)", "averaged", "results", algorithmNames, datasets);
	}
}