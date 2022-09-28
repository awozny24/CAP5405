import React from "react";
import Column from "./column";

class Board extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      player: "x",
      board: [
        ["b", "b", "b", "b", "b", "b", "b"],
        ["b", "b", "b", "b", "b", "b", "b"],
        ["b", "b", "b", "b", "b", "b", "b"],
        ["b", "b", "b", "b", "b", "b", "b"],
        ["b", "b", "b", "b", "b", "b", "b"],
        ["b", "b", "b", "b", "b", "b", "b"],
      ],
    };
  }

  transpose = (arr) => {
    for (let i = 0; i < arr.length; i++) {
      for (let j = 0; j < i; j++) {
        const tmp = arr[i][j];
        arr[i][j] = arr[j][i];
        arr[j][i] = tmp;
      }
    }
    return arr;
  };

  currBoardState = [[], [], [], [], [], [], []];

  boardStateHandler = (columnState, columnNo) => {
    let currColumnState = [...columnState];
    while (currColumnState.length < 6) {
      currColumnState.push("b");
    }
    this.currBoardState[columnNo] = currColumnState;
    let transpose = [
      ["b", "b", "b", "b", "b", "b", "b"],
      ["b", "b", "b", "b", "b", "b", "b"],
      ["b", "b", "b", "b", "b", "b", "b"],
      ["b", "b", "b", "b", "b", "b", "b"],
      ["b", "b", "b", "b", "b", "b", "b"],
      ["b", "b", "b", "b", "b", "b", "b"],
    ];
    for (let i = 0; i < 7; i++) {
      for (let j = 0; j < 6; j++) {
        transpose[j][i] = this.currBoardState[i][j];
      }
    }
    transpose = transpose.reverse();
    if (this.state.board === transpose) {
      this.setState({ board: transpose });
    }
    // this.setState({player:"b"});
  };

  onBoardClickHandler = () => {
    if (this.state.player == "x") {
      this.setState({ player: "o" });
    } else {
      this.setState({ player: "x" });
    }
  };

  render() {
    return (
      <div className="board">
        <Column
          player={this.state.player}
          onBoardClickHandler={this.onBoardClickHandler}
          boardStateHandler={this.boardStateHandler}
          columnNo={0}
        ></Column>
        <Column
          player={this.state.player}
          onBoardClickHandler={this.onBoardClickHandler}
          boardStateHandler={this.boardStateHandler}
          columnNo={1}
        ></Column>
        <Column
          player={this.state.player}
          onBoardClickHandler={this.onBoardClickHandler}
          boardStateHandler={this.boardStateHandler}
          columnNo={2}
        ></Column>
        <Column
          player={this.state.player}
          onBoardClickHandler={this.onBoardClickHandler}
          boardStateHandler={this.boardStateHandler}
          columnNo={3}
        ></Column>
        <Column
          player={this.state.player}
          onBoardClickHandler={this.onBoardClickHandler}
          boardStateHandler={this.boardStateHandler}
          columnNo={4}
        ></Column>
        <Column
          player={this.state.player}
          onBoardClickHandler={this.onBoardClickHandler}
          boardStateHandler={this.boardStateHandler}
          columnNo={5}
        ></Column>
        <Column
          player={this.state.player}
          onBoardClickHandler={this.onBoardClickHandler}
          boardStateHandler={this.boardStateHandler}
          columnNo={6}
        ></Column>
      </div>
    );
  }
}

export default Board;
