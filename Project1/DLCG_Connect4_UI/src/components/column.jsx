import React from "react";
import Block from "./block";

class Column extends React.Component {
  constructor(props) {
    super(props);
    this.state = { filledblock: 0 };
  }

  onClickHandler = () => {
    if (this.state.filledblock < 6) {
      this.setState({
        filledblock: this.state.filledblock + 1,
      });

      this.props.boardStateHandler(this.props.columnNo, this.state.filledblock);
      this.props.onBoardClickHandler();
    } else {
      this.props.alertfunc("Invalid Move");
    }
  };

  setblocks = () => {
    let blocks = [];
    for (let i in this.props.blockSequence) {
      if (this.props.blockSequence[i] === 0) {
        blocks.push(<Block key={i} filled={false} player="b" />);
      } else if (this.props.blockSequence[i] === 1) {
        blocks.push(<Block key={i} filled={true} player="x" />);
      } else {
        blocks.push(<Block key={i} filled={true} player="o" />);
      }
    }
    return blocks.reverse();
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
