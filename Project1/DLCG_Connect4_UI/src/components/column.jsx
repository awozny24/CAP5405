import React from "react";
import Block from "./block";

class Column extends React.Component {
  constructor(props) {
    super(props);
    this.state = { filledblock: 0, filledblockplayers: [] };
  }

  onClickHandler = () => {
    if (this.state.filledblockplayers.length < 6) {
      this.setState({
        filledblock: this.state.filledblock + 1,
        filledblockplayers:
          this.state.filledblockplayers.length < 6
            ? [...this.state.filledblockplayers, this.props.player]
            : this.state.filledblockplayers,
      });
      this.props.onBoardClickHandler();
    } else {
      alert("Invalid Move");
    }
  };

  setblocks = () => {
    let blocks = [];
    for (let i = 6; i > 0; i--) {
      if (i <= this.state.filledblock) {
        blocks.push(
          <Block
            key={i}
            filled={true}
            player={this.state.filledblockplayers[i - 1]}
          ></Block>
        );
      } else {
        blocks.push(<Block key={i} filled={false} player={"b"}></Block>);
      }
    }
    this.props.boardStateHandler(
      this.state.filledblockplayers,
      this.props.columnNo
    );
    return blocks;
  };

  render() {
    return (
      <div className="column" onClick={this.onClickHandler}>
        {this.setblocks()}
      </div>
    );
  }
}

export default Column;
