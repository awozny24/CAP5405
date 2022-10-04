import React from "react";
import Column from "./column";
import * as tf from "@tensorflow/tfjs";
import GameOver from "./gameover";

class Board extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      player: "x",
      0: [0, 0, 0, 0, 0, 0],
      1: [0, 0, 0, 0, 0, 0],
      2: [0, 0, 0, 0, 0, 0],
      3: [0, 0, 0, 0, 0, 0],
      4: [0, 0, 0, 0, 0, 0],
      5: [0, 0, 0, 0, 0, 0],
      6: [0, 0, 0, 0, 0, 0],
      gameState: "Draw",
      noOfMoves: 0,
      gameOver: false,
    };
  }

  onBoardClickHandler = () => {
    if (this.state.player == "x") {
      this.setState({ player: "o" });
    } else {
      this.setState({ player: "x" });
    }
  };

  boardStateHandler = (columnNo, filledblock) => {
    let colState = [];
    for (let i in this.state[columnNo]) {
      if (i == filledblock) {
        if (this.state.player == "x") {
          colState.push(1);
        } else {
          colState.push(-1);
        }
      } else {
        colState.push(this.state[columnNo][i]);
      }
    }
    let res = "";
    this.setState(
      (prevState) => {
        let newState = {};
        for (let i in Object.keys(prevState)) {
          if (Object.keys(prevState)[i] == columnNo) {
            newState[columnNo] = colState;
          } else {
            newState[Object.keys(prevState)[i]] =
              prevState[Object.keys(prevState)[i]];
          }
        }
        return newState;
      },
      async () => {
        let arr = [
          [...this.state[0]],
          [...this.state[1]],
          [...this.state[2]],
          [...this.state[3]],
          [...this.state[4]],
          [...this.state[5]],
          [...this.state[6]],
        ];
        if (this.state.noOfMoves >= 6) {
          res = await this.props.predictfunc(arr);
        } else {
          res = 0;
        }
        this.setState((prevState) => {
          let gameOverState = false;
          let gameNewState = "";
          if (res == 0) {
            gameNewState = "Draw";
          } else if (res == 1) {
            gameNewState = "X Wins";
            gameOverState = true;
          } else {
            gameNewState = "X Loses";
            gameOverState = true;
          }
          return {
            ...prevState,
            gameState: gameNewState,
            noOfMoves: this.state.noOfMoves + 1,
            gameOver: gameOverState,
          };
        });
      }
    );
    console.log(this.state.noOfMoves);
    return this.state.gameState;
  };

  // playAgain = () => {
  //   this.setState((prevState) => {
  //     return {
  //       ...prevState,
  //       gameOver: false,
  //       0: [0, 0, 0, 0, 0, 0],
  //       1: [0, 0, 0, 0, 0, 0],
  //       2: [0, 0, 0, 0, 0, 0],
  //       3: [0, 0, 0, 0, 0, 0],
  //       4: [0, 0, 0, 0, 0, 0],
  //       5: [0, 0, 0, 0, 0, 0],
  //       6: [0, 0, 0, 0, 0, 0],
  //     };
  //   });

  // };

  render() {
    return (
      <div>
        <div className="board">
          {this.state.gameOver ? (
            <GameOver
              result={this.state.gameState}
              playAgainfunc={this.playAgain}
            />
          ) : (
            <div></div>
          )}
          <div className="status">
            <div className="player">{this.state.player}'s Turn</div>
            <div className="match">{this.state.gameState}</div>
          </div>
          <Column
            player={this.state.player}
            onBoardClickHandler={this.onBoardClickHandler}
            boardStateHandler={this.boardStateHandler}
            alertfunc={this.props.alertfunc}
            blockSequence={this.state[0]}
            columnNo={0}
          ></Column>
          <Column
            player={this.state.player}
            onBoardClickHandler={this.onBoardClickHandler}
            boardStateHandler={this.boardStateHandler}
            alertfunc={this.props.alertfunc}
            blockSequence={this.state[1]}
            columnNo={1}
          ></Column>
          <Column
            player={this.state.player}
            onBoardClickHandler={this.onBoardClickHandler}
            boardStateHandler={this.boardStateHandler}
            alertfunc={this.props.alertfunc}
            blockSequence={this.state[2]}
            columnNo={2}
          ></Column>
          <Column
            player={this.state.player}
            onBoardClickHandler={this.onBoardClickHandler}
            boardStateHandler={this.boardStateHandler}
            alertfunc={this.props.alertfunc}
            blockSequence={this.state[3]}
            columnNo={3}
          ></Column>
          <Column
            player={this.state.player}
            onBoardClickHandler={this.onBoardClickHandler}
            boardStateHandler={this.boardStateHandler}
            alertfunc={this.props.alertfunc}
            blockSequence={this.state[4]}
            columnNo={4}
          ></Column>
          <Column
            player={this.state.player}
            onBoardClickHandler={this.onBoardClickHandler}
            boardStateHandler={this.boardStateHandler}
            alertfunc={this.props.alertfunc}
            blockSequence={this.state[5]}
            columnNo={5}
          ></Column>
          <Column
            player={this.state.player}
            onBoardClickHandler={this.onBoardClickHandler}
            boardStateHandler={this.boardStateHandler}
            alertfunc={this.props.alertfunc}
            blockSequence={this.state[6]}
            columnNo={6}
          ></Column>
        </div>
      </div>
    );
  }
}

export default Board;
