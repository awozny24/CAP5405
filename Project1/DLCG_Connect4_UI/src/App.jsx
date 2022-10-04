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
    // let prediction = this.model.predict(tf.tensor2d(arr, [1, 42]));
    // let predictionObject = prediction.arraySync()[0];
    // let res = 0;
    // for (let i in predictionObject) {
    //   if (predictionObject[i] > predictionObject[res]) {
    //     res = i;
    //   }
    // }
    // console.log(predictionObject);
    // return res;
    let transpose = [
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0],
    ];
    let full = true;
    for(let i = 0; i < 6; i++){
      for(let j = 0; j < 7; j++){
        if(transpose[i][j] == 0){
          full = false;
          break;
        }
      }
    }
    if(full){
      return 4;
    }

    for(let i = 0; i < 6; i++){
      for(let j = 0; j < 7; j++){
        // console.log(i,j)
        transpose[i][j] = arr[j][i];
      }
    }
    transpose = transpose.reverse();
     // Checking Win Horizontanlly
     for(let i in transpose){
      for(let j = 0; j < 3; j++){
        if(transpose[i][j] == 1 && transpose[i][j+1] == 1 && transpose[i][j+2] == 1 && transpose[i][j+3] == 1){
          return 1;
        }else if(transpose[i][j] == -1 && transpose[i][j+1] == -1 && transpose[i][j+2] == -1 && transpose[i][j+3] == -1){
          return 2;
        }
      }
     }

     // Checking Win Vertically
     for(let i in arr){
      for(let j = 0; j < 2; j++){
        if(arr[i][j] == 1 && arr[i][j+1] == 1 && arr[i][j+2] == 1 && arr[i][j+3] == 1){
          return 1;
        }else if(arr[i][j] == -1 && arr[i][j+1] == -1 && arr[i][j+2] == -1 && arr[i][j+3] == -1){
          return 2;
        }
      }
     }

     // Checking Win Diagonally 1
     for(let i = 0; i < 3; i++){
      for(let j = 0; j < 4; j++){
        if(transpose[i][j] == 1 && transpose[i+1][j+1] == 1 && transpose[i+2][j+2] == 1 && transpose[i+3][j+3] == 1){
          return 1;
        }else if(transpose[i][j] == -1 && transpose[i+1][j+1] == -1 && transpose[i+2][j+2] == -1 && transpose[i+3][j+3] == -1){
          return 2;
        }
      }
     }

     // Checking Win Diagonally 2
     for(let i = 0; i < 3; i++){
      for(let j = 6; j > 2; j--){
        if(transpose[i][j] == 1 && transpose[i+1][j-1] == 1 && transpose[i+2][j-2] == 1 && transpose[i+3][j-3] == 1){
          return 1;
        }else if(transpose[i][j] == -1 && transpose[i+1][j-1] == -1 && transpose[i+2][j-2] == -1 && transpose[i+3][j-3] == -1){
          return 2;
        }
      }
     }

    return 0;
  };
  async componentDidMount() {
    // this.model = await tf.loadLayersModel("./tfjsmodel/model.json");
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
