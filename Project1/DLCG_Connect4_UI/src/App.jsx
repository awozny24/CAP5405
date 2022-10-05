import React from "react";
import "./App.css";
import Board from "./components/board";
import Modal from "./components/modal";
import * as tf from "@tensorflow/tfjs";
import Select1or2 from "./components/select1or2";

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = { modal: [], player1or2: 0, playerSelect: [] };
    this.child = React.createRef();
  }

  predict = (arr) => {
    let prediction = this.model.predict(tf.tensor2d(arr, [1, 42]));
    let predictionObject = prediction.arraySync()[0];
    let res = 0;
    for (let i in predictionObject) {
      if (predictionObject[i] > predictionObject[res]) {
        res = i;
      }
    }
    return [res, predictionObject[res]];
  };

  checkWin = (arr) => {
    let transpose = [
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0],
    ];

    for (let i = 0; i < 6; i++) {
      for (let j = 0; j < 7; j++) {
        transpose[i][j] = arr[j][i];
      }
    }
    transpose = transpose.reverse();
    // Checking Win Horizontanlly
    for (let i in transpose) {
      for (let j = 0; j < 4; j++) {
        if (
          transpose[i][j] == 1 &&
          transpose[i][j + 1] == 1 &&
          transpose[i][j + 2] == 1 &&
          transpose[i][j + 3] == 1
        ) {
          return 1;
        } else if (
          transpose[i][j] == -1 &&
          transpose[i][j + 1] == -1 &&
          transpose[i][j + 2] == -1 &&
          transpose[i][j + 3] == -1
        ) {
          return 2;
        }
      }
    }

    // Checking Win Vertically
    for (let i in arr) {
      for (let j = 0; j < 3; j++) {
        if (
          arr[i][j] == 1 &&
          arr[i][j + 1] == 1 &&
          arr[i][j + 2] == 1 &&
          arr[i][j + 3] == 1
        ) {
          return 1;
        } else if (
          arr[i][j] == -1 &&
          arr[i][j + 1] == -1 &&
          arr[i][j + 2] == -1 &&
          arr[i][j + 3] == -1
        ) {
          return 2;
        }
      }
    }

    // Checking Win Diagonally 1
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 4; j++) {
        if (
          transpose[i][j] == 1 &&
          transpose[i + 1][j + 1] == 1 &&
          transpose[i + 2][j + 2] == 1 &&
          transpose[i + 3][j + 3] == 1
        ) {
          return 1;
        } else if (
          transpose[i][j] == -1 &&
          transpose[i + 1][j + 1] == -1 &&
          transpose[i + 2][j + 2] == -1 &&
          transpose[i + 3][j + 3] == -1
        ) {
          return 2;
        }
      }
    }

    // Checking Win Diagonally 2
    for (let i = 0; i < 3; i++) {
      for (let j = 6; j > 2; j--) {
        if (
          transpose[i][j] == 1 &&
          transpose[i + 1][j - 1] == 1 &&
          transpose[i + 2][j - 2] == 1 &&
          transpose[i + 3][j - 3] == 1
        ) {
          return 1;
        } else if (
          transpose[i][j] == -1 &&
          transpose[i + 1][j - 1] == -1 &&
          transpose[i + 2][j - 2] == -1 &&
          transpose[i + 3][j - 3] == -1
        ) {
          return 2;
        }
      }
    }

    let full = true;
    for (let i = 0; i < 6; i++) {
      for (let j = 0; j < 7; j++) {
        if (transpose[i][j] == 0) {
          full = false;
          break;
        }
      }
    }
    if (full) {
      return 4;
    }

    return 0;
  };

  player1Func = () => {
    this.child.current.firstPlay();
    this.setState((prevState) => {
      return { ...prevState, player1or2: 1, playerSelect: [] };
    });
  };

  player2Func = () => {
    this.setState((prevState) => {
      return { ...prevState, player1or2: 2, playerSelect: [] };
    });
  };
  getRandomInt = (max) => {
    return Math.floor(Math.random() * max);
  };
  getNextMove = (arr) => {
    // all possible moves
    let possiblemoves = [];
    let tempNewColumns = [];
    for (let i in arr) {
      let tempNewColumn = [];
      let done = false;
      for (let j in arr[i]) {
        if (!done && arr[i][j] === 0) {
          tempNewColumn.push(1);
          done = true;
        } else {
          tempNewColumn.push(arr[i][j]);
        }
      }
      tempNewColumns.push(tempNewColumn);
    }

    for (let i in tempNewColumns) {
      let move = [];
      for (let j = 0; j < 7; j++) {
        if (i == j) {
          move.push(tempNewColumns[i]);
        } else {
          move.push(arr[j]);
        }
      }
      possiblemoves.push(move);
    }

    // Checking for Obvious wins
    for (let i in possiblemoves) {
      let r = this.checkWin(possiblemoves[i]);
      if (r === 1) {
        return i;
      }
    }

    // Checking wins by model
    let predictions = [];
    let predscore = [];
    for (let i in possiblemoves) {
      let prediction = this.predict([
        ...possiblemoves[i][0],
        ...possiblemoves[i][1],
        ...possiblemoves[i][2],
        ...possiblemoves[i][3],
        ...possiblemoves[i][4],
        ...possiblemoves[i][5],
        ...possiblemoves[i][6],
      ]);
      if (prediction[0] === "1") {
        predictions.push(i);
        predscore.push(prediction[1]);
      }
    }
    if (predictions.length === 0) {
      return this.getRandomInt(7);
    } else if (predictions.length === 1) {
      return parseInt(predictions[0]);
    } else {
      let maxpredscore = predscore[0];
      let maxpred = predictions[0];
      for (let i in predscore) {
        if (maxpredscore < predscore[i]) {
          maxpredscore = predscore[i];
          maxpred = predictions[i];
        }
      }
      return parseInt(maxpred);
    }
  };

  async componentDidMount() {
    this.model = await tf.loadLayersModel("./tfjsmodel/model.json");
    this.setState((prevState) => {
      return {
        ...prevState,
        playerSelect: [
          <Select1or2
            player1={this.player1Func}
            player2={this.player2Func}
            key="247878968"
          />,
        ],
      };
    });
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
        {this.state.playerSelect}
        {this.state.modal}
        <h1>CONNECT 4</h1>
        <Board
          ref={this.child}
          alertfunc={this.alertModal}
          predictfunc={this.checkWin}
          players={this.state.player1or2}
          getNextMove={this.getNextMove}
        />
      </div>
    );
  }
}

export default App;
