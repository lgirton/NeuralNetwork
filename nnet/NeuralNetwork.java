package nnet;

import java.io.OutputStream;
import java.io.PrintStream;
import java.util.Random;

public class NeuralNetwork {

	static class Environment {

		private static Random random = null;

		public static synchronized Random getRandom() {
			if (random == null) {
				random = new Random();
			}
			return random;
		}


	}
	
	public static void dump(double[][] matrix) {
		for (int m = 0; m < matrix.length; m++) {
			for (int n = 0; n < matrix[m].length; n++) {
				System.out.print(String.format("%.5f\t", matrix[m][n]));
			}
			System.out.println();
		}
		System.out.println();
	}

	public static void dump(double[] vector) {
		for (int i = 0; i < vector.length; i++) {
			System.out.println(String.format("%.5f", vector[i]));
		}
		System.out.println();
	}


	static interface Source {
		public double[] getOutput();
	}

	static class InputSource implements Source {
		private final double[] output;

		public InputSource(final int numInputs) {
			output = new double[numInputs];
		}

		@Override
		public double[] getOutput() {
			return output;
		}

		public void setOutput(double[] output) {
			System.arraycopy(output, 0, this.output, 0, output.length);
		}

	}

	static abstract class Layer implements Source {
		protected Source source;

		protected final double[] output;
		protected final double[] error;
		protected final double[][] weight;

		public Layer(final int numNodes) {
			output = new double[numNodes];
			error = new double[numNodes];
			weight = new double[numNodes][];
		}

		public double[] getOutput() {
			double[] dest = new double[output.length];
			System.arraycopy(output, 0, dest, 0, output.length);
			return dest;
		}

		public void setSource(Source source) {
			this.source = source;
			initializeWeights();
		}

		private void initializeWeights() {
			final int numInputs = source.getOutput().length;
			for (int i = 0; i < weight.length; i++) {
				weight[i] = new double[numInputs];
				for (int j = 0; j < numInputs; j++) {
					weight[i][j] = Environment.getRandom().nextDouble() - 0.5;
				}
			}
		}

		public void feedforward() {
			double[] inputs = source.getOutput();
			double[] sigmoid = sigmoid(innerProduct(weight, inputs));
			System.arraycopy(sigmoid, 0, output, 0, output.length);
		}

		protected void updateWeight() {
			double[] input = source.getOutput();
			for (int i = 0; i < error.length; i++) {
				for (int j = 0; j < input.length; j++) {
					weight[i][j] = weight[i][j] + (error[i] * input[j]);
				}
			}
		}

		double[] sigmoid(double[] x) {
			double[] sigmoid = new double[x.length];
			for (int i = 0; i < x.length; i++) {
				sigmoid[i] = sigmoid(x[i]);
			}
			return sigmoid;
		}

		double sigmoid(double x) {
			return 1 / (1 + Math.exp(-x));
		}

		double[] innerProduct(double[][] A, double[] x) {
			double[] product = new double[A.length];
			for (int m = 0; m < A.length; m++) {
				for (int n = 0; n < x.length; n++) {
					product[m] += A[m][n] * x[n];
				}
			}
			return product;
		}

		double[][] transpose(double[][] A) {
			double[][] t = new double[A[0].length][A.length];
			for (int m = 0; m < t.length; m++) {
				for (int n = 0; n < t[m].length; n++) {
					t[m][n] = A[n][m];
				}
			}
			return t;
		}
		
		public void dump(OutputStream out) {
			PrintStream printer = new PrintStream(out);
			printer.println("Hello World");
			
			
			printer.flush();
		}

	}

	static class HiddenLayer extends Layer {

		Layer sink;

		public HiddenLayer(int numNodes) {
			super(numNodes);
		}

		public void backpropagate() {
			calculateError();
			updateWeight();
		}

		private void calculateError() {
			double[] product = innerProduct(transpose(sink.weight), sink.error);
			for (int i = 0; i < error.length; i++) {
				double o = output[i];
				error[i] = o * (1 - o) * product[i];
			}

		}

		public void setSink(Layer sink) {
			this.sink = sink;

		}
	}

	static class OutputLayer extends Layer {

		public OutputLayer(int numNodes) {
			super(numNodes);
		}

		public void backpropagate(double[] target) {
			calculateError(target);
			updateWeight();
		}

		private void calculateError(double[] target) {
			for (int i = 0; i < error.length; i++) {
				double o = this.output[i];
				error[i] = o * (1 - o) * (target[i] - o);
			}
		}

	}

	public static void main(String[] args) {
		InputSource input = new InputSource(2);

		OutputLayer output = new OutputLayer(1);

		HiddenLayer hidden = new HiddenLayer(2);
		hidden.setSource(input);
		hidden.setSink(output);

		hidden.weight[0][0] = 0.1;
		hidden.weight[0][1] = 0.8;
		hidden.weight[1][0] = 0.4;
		hidden.weight[1][1] = 0.6;

		output.setSource(hidden);
		output.weight[0][0] = 0.3;
		output.weight[0][1] = 0.9;

		input.setOutput(new double[] { 0.35, 0.9 });

		hidden.feedforward();

		System.out.println("Input Layer: ");
		dump(input.getOutput());

		System.out.println("Hidden Layer Weights: ");
		dump(hidden.weight);

		System.out.println("Hidden Layer Output: ");
		dump(hidden.getOutput());

		output.feedforward();

		System.out.println("Output Layer Weights: ");
		dump(output.weight);

		System.out.println("Output Layer Output: ");
		dump(output.getOutput());

		output.backpropagate(new double[] { 0.5 });

		System.out.println("Output Layer Error: ");
		dump(output.error);

		System.out.println("Output Weights After Update: ");
		dump(output.weight);

		hidden.backpropagate();

		System.out.println("Hidden Layer Error: ");
		dump(hidden.error);

		System.out.println("Hidden Layer Updated Weights: ");
		dump(hidden.weight);
		
		hidden.dump(System.out);
	}

}
