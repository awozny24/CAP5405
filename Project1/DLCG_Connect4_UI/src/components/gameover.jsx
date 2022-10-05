import React from "react";
import "../App.css";

class GameOver extends React.Component{
    render(){
        return <div className="game-over">
            <div className="pop-up end-pop-up" style={{transformY:"200px"}}>
                <h1>Game Over!</h1>
                <br/>
                <h2>{this.props.result}</h2>
                <button className="play-again-btn" onClick={this.props.playAgainfunc}>Play Again</button>
            </div>
        </div>
    }
}

export default GameOver;