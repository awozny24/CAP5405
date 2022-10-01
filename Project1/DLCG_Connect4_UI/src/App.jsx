import React from "react";
import "./App.css";
import Board from "./components/board";
import Modal from "./components/modal";
import * as tf from "@tensorflow/tfjs";

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = { modal: [] };
  }
  predict = async (arr) => {
    // this.model = await tf.loadLayersModel("./tfjsmodel/model.json");
    let prediction = this.model.predict(tf.tensor2d(arr, [1, 42]));
    let predictionObject = prediction.arraySync()[0];
    let res = 0;
    for (let i in predictionObject) {
      if (predictionObject[i] > predictionObject[res]) {
        res = i;
      }
    }
    console.log(predictionObject);
    return res;
  };
  async componentDidMount() {
    this.model = await tf.loadLayersModel("./tfjsmodel/model.json");
  }
  alertModal = (title) => {
    this.setState({ modal: [<Modal key={title} title={title} />] });
    setTimeout(() => {
      this.setState({ modal: [] });
    }, 2000);
  };
  render() {
    return (
      <div className="App">
        {this.state.modal}
        <h1>CONNECT 4</h1>
        <Board alertfunc={this.alertModal} predictfunc={this.predict} />
      </div>
    );
  }
}

export default App;
