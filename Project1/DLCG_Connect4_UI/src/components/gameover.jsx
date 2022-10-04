import React from "react";
import "../App.css";

class GameOver extends React.Component{
    render(){
        return <div className="game-over">
            <div className="pop-up">
                <h1>Game Over!</h1>
                <br/>
                <h2>{this.props.result}</h2>
                <button className="play-again-btn" onClick={() => window.location.reload()}>Play Again</button>
            </div>
        </div>
    }
}

export default GameOver;