import React from "react";
import "../App.css";

class Select1or2 extends React.Component{
    render(){
        return <div className="game-over">
            <div className="pop-up">
                <h1>New Game!</h1>
                <br/>
                <h2>Select Any option : </h2>
                <h4>1 player - AI as Player 1</h4>
                <h4>2 players</h4>
                <button className="play-again-btn" onClick={this.props.player1}>1 Player</button>
                <button className="play-again-btn" onClick={this.props.player2}>2 Players</button>
            </div>
        </div>
    }
}

export default Select1or2;