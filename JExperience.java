import javax.swing.SwingUtilities;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JButton;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;
import javax.swing.Timer;



public class JExperience {
  public static void main(String[] args) {
    // GUIを構築する処理を専用のスレッド内で呼び出す．
  	// ※Swingはシングルスレッド設計であるため，
  	// Frameを描画する処理をイベントディスパッチスレッドで
  	// 実行させることが推奨されている．
  	SwingUtilities.invokeLater(new Runnable() {
      @Override
      public void run() {
        JExperience exp = new JExperience();
        exp.exec();
      }
    });
  }

  public void exec() {
    // ウィンドウを作成する．
    final JFrame frame = new JFrame("Java Experience");

  	// 特徴ベクトルを読み込む．
    Dataset dataset = CsvDatasetReader.readDataset("scurve.csv");
    // Dataset dataset = CsvDatasetReader.readDataset("Iris_data.csv");

    // 描画画面を作成する．
  	// final JPanel panel = new JPanel();
  	final VPanel panel = new VPanel();
  	// panel.setPreferredSize(new Dimension(300, 300));

    // 埋込法（マッピングモデル）を定義する．
    // final EmbeddingMethod embMethod = new MDS();
    // final EmbeddingMethod embMethod = new NLM();
    final EmbeddingMethod embMethod = new SNE();
    // マッピングモデルに特徴ベクトルをセットする．
    embMethod.setDataset(dataset);

    // マッピングモデルのオブサーバーとして描画画面を登録する．
    embMethod.setVisualizer(panel);

    // タイマーを作成する．
    final Timer timer = new Timer(50, new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        embMethod.update();
      }
    });

    // ボタンを作成（追加）する．
    final JPanel panelButton = new JPanel();
    final JButton buttonRun = new JButton("Run");
    final JButton buttonStop = new JButton("Stop");
    panelButton.add(buttonRun);
    panelButton.add(buttonStop);


  	// RUNボタンに動作を登録（定義）する．
    buttonRun.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
		// パネルの背景を変更する
        panel.setBackground(Color.BLUE);
		// 最適化スタート
        embMethod.init();
        timer.start();
      }
    });


    buttonStop.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        //最適化ストップ
        timer.stop();
      }
    });


    // ウィンドウのレイアウトを設定する．
    final Container content = frame.getContentPane();
    content.setLayout(new BorderLayout());

    // 作成したコンポーネントを追加する．
  	content.add(panel, BorderLayout.PAGE_START);
    content.add(panelButton, BorderLayout.PAGE_END);

    // ウィンドウのサイズをコンテンツのサイズに合わせる．
    frame.pack();

  	// // ウインドウのサイズを固定する．
  	// frame.setResizable(false);
  	
    // ウィンドウの位置をスクリーンの中央に指定する．
    frame.setLocationRelativeTo(null);

    // ウィンドウの閉じるボタンが押されたらプログラムを終了する．
    frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

    // ウィンドウを表示する．
    frame.setResizable(false);
    frame.setVisible(true);
  }
}
