import React from "react";
import "./../App.css";

class Modal extends React.Component {
  constructor(props) {
    super(props);
    this.state = { show: true };
  }

  render() {
    return (
      <div className="modal">
        {this.props.title}
        <div className="loader"></div>
      </div>
    );
  }
}

export default Modal;
