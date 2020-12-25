import java.lang.Math;
import java.util.Arrays;

public class SNE extends MDS {
  protected double[][] highTransitionProbabilityMatrix;
  protected double[][] lowTransitionProbabilityMatrix;

  SNE() {
    this.isInitialized = false;
    this.learningRate = 0.1;
  }

  protected void calculateProbabilityMatrix(double[][] distanceMatrix, double[][] probabilityMatrix, double c) {
    int numSamples = distanceMatrix.length;
    for (int i = 0; i < numSamples; i++) {
      for (int j = i + 1; j < numSamples; j++) {
        double similarity = Math.exp(-(distanceMatrix[i][j] * distanceMatrix[i][j]) / c);
        probabilityMatrix[i][j] = similarity;
        probabilityMatrix[j][i] = similarity;
      }
      probabilityMatrix[i][i] = 0.0;
    }
    for (int i = 0; i < numSamples; i++) {
      double degree = 0.0;
      for (int j = 0; j < numSamples; j++) {
        degree += probabilityMatrix[i][j];
      }
      for (int j = 0; j < numSamples; j++) {
        probabilityMatrix[i][j] /= degree;
      }
    }
  }

  @Override
  public void init(){
    this.isInitialized = false;
    if (this.highDataset.isEmpty() == false) {
      // サンプル数を得る．
      int numSamples = this.highDataset.getNumSamples();
      // 低次元空間を初期化する．
      double[][] lowFeatures = new double[numSamples][2];
      String[] lowLabels = new String[numSamples];
      for (int i = 0; i < numSamples; i++) {
        lowFeatures[i][0] = Math.random();
        lowFeatures[i][1] = Math.random();
        lowLabels[i] = this.highDataset.getLabel(i);
      }
      this.lowDataset = new Dataset(lowLabels, lowFeatures);
      // 高次元空間の距離行列を計算しておく．
      this.highDistanceMatrix = new double[numSamples][numSamples];
      this.lowDistanceMatrix = new double[numSamples][numSamples];
      this.highTransitionProbabilityMatrix = new double[numSamples][numSamples];
      this.lowTransitionProbabilityMatrix = new double[numSamples][numSamples];

      this.calculateDistanceMatrix(this.highDataset, this.highDistanceMatrix);
      this.calculateProbabilityMatrix(this.highDistanceMatrix, this.highTransitionProbabilityMatrix, 2.0);
      // 初期化を完了した．
      this.isInitialized = true;
    }
  }

  @Override
  public void update() {
    if (this.isInitialized == true) {
      // 低次元空間の距離行列を計算する．
      this.calculateDistanceMatrix(this.lowDataset, this.lowDistanceMatrix);
      this.calculateProbabilityMatrix(this.lowDistanceMatrix, this.lowTransitionProbabilityMatrix,1.0);
      // 勾配を計算する．
      int numSamples = this.highDataset.getNumSamples();
      double[][] gradients = new double[numSamples][2];
      for (int i = 0; i < numSamples; i++) {
        Arrays.fill(gradients[i], 0.0); // 0で勾配を初期化する．
      	// 式(2)を参考にして勾配計算を実装せよ．
      	for(int j=0;j<numSamples;j++){
          if(i!=j){
            double coeff = 2*(this.highTransitionProbabilityMatrix[i][j]-this.lowTransitionProbabilityMatrix[i][j]+
                              this.highTransitionProbabilityMatrix[j][i]-this.lowTransitionProbabilityMatrix[j][i]);
            gradients[i][0] += coeff*(this.lowDataset.getFeature(i,0)-this.lowDataset.getFeature(j,0));
            gradients[i][1] += coeff*(this.lowDataset.getFeature(i,1)-this.lowDataset.getFeature(j,1));
          }
        }
      }
      // 低次元空間の座標を更新する．
      for (int i = 0; i < numSamples; i++) {
      	// 式(3)を参考にして低次元空間の更新を実装する．
          this.lowDataset.updateFeature(i,0,lowDataset.getFeature(i,0)-learningRate*gradients[i][0]);
          this.lowDataset.updateFeature(i,1,lowDataset.getFeature(i,1)-learningRate*gradients[i][1]);
          this.lowDataset.updateLabel(i,this.lowDataset.getLabel(i)); 
      }
      // ビジュアライザーに低次元空間の座標の更新を伝える．
      this.notifyVisualizer();
    }
  }

}
