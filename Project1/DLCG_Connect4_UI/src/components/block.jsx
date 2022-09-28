import React from "react";

class Block extends React.Component {
  render() {
    return (
      <div className="block">
        <div
          className={
            this.props.filled
              ? this.props.player == "x"
                ? "circlex"
                : "circleo"
              : "circle"
          }
        ></div>
      </div>
    );
  }
}

export default Block;
