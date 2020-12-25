import java.lang.Math;
import java.util.Arrays;

public class NLM extends MDS {
  protected boolean isInitialized;
  protected double learningRate;
  protected double[][] highDistanceMatrix;
  protected double[][] lowDistanceMatrix;
  
  NLM() {
    this.isInitialized = false;
    this.learningRate = 0.001;
  }

  

  @Override
  public void update() {
    if (this.isInitialized == true) {
      // 低次元空間の距離行列を計算する．
      this.calculateDistanceMatrix(this.lowDataset, this.lowDistanceMatrix);
      // 勾配を計算する．
      int numSamples = this.highDataset.getNumSamples();
      double[][] gradients = new double[numSamples][2];
      for (int i = 0; i < numSamples; i++) {
        Arrays.fill(gradients[i], 0.0); // 0で勾配を初期化する．
      	// 式(2)を参考にして勾配計算を実装せよ．
      	for(int j = 0; j < numSamples;j++){
          if(i != j){
            double coeff = (this.highDistanceMatrix[i][j]-this.lowDistanceMatrix[i][j])/(this.highDistanceMatrix[i][j]*this.lowDistanceMatrix[i][j]);
            gradients[i][0] += coeff*(this.lowDataset.getFeature(j,0)-this.lowDataset.getFeature(i,0));
            gradients[i][1] += coeff*(this.lowDataset.getFeature(j,1)-this.lowDataset.getFeature(i,1));
          }
        }
      }
      // 低次元空間の座標を更新する．
      for (int i = 0; i < numSamples; i++) {
        // 式(3)を参考にして低次元空間の更新を実装せよ．
        this.lowDataset.updateFeature(i,0,this.lowDataset.getFeature(i,0)-learningRate*gradients[i][0]); 
        this.lowDataset.updateFeature(i,1,this.lowDataset.getFeature(i,1)-learningRate*gradients[i][1]); 
        this.lowDataset.updateLabel(i,this.lowDataset.getLabel(i)); 
      }
      // ビジュアライザーに低次元空間の座標の更新を伝える．
      this.notifyVisualizer();
    }
  }
}
