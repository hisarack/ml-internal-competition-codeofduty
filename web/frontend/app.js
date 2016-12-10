var DigitBlock = React.createClass({
  render: function(){

    color0 = "black";
    if(this.props.digit.is0 > 0.5){
      color0 = "red";
    }
    color1 = "black";
    if(this.props.digit.is1 > 0.5){
      color1 = "red";
    }
    color2 = "black";
    if(this.props.digit.is2 > 0.5){
      color2 = "red";
    }
    color3 = "black";
    if(this.props.digit.is3 > 0.5){
      color3 = "red";
    }
    color4 = "black";
    if(this.props.digit.is4 > 0.5){
      color4 = "red";
    }
    color5 = "black";
    if(this.props.digit.is5 > 0.5){
      color5 = "red";
    }
    color6 = "black";
    if(this.props.digit.is6 > 0.5){
      color6 = "red";
    }
    color7 = "black";
    if(this.props.digit.is7 > 0.5){
      color7 = "red";
    }
    color8 = "black";
    if(this.props.digit.is8 > 0.5){
      color8 = "red";
    }
    color9 = "black";
    if(this.props.digit.is9 > 0.5){
      color9 = "red";
    }


    return (
      <tr>
        <td><img src={"./digit/"+this.props.digit.fname+".png"} style={{width:"50px", height:"50px"}}/></td>
        <td><img src={"./digit2/"+this.props.digit.fname+".png"} style={{width:"50px", height:"50px"}}/></td>
        <td style={{color:color0}}>{"0:"+this.props.digit.is0}</td>
        <td style={{color:color1}}>{"1:"+this.props.digit.is1}</td>
        <td style={{color:color2}}>{"2:"+this.props.digit.is2}</td>
        <td style={{color:color3}}>{"3:"+this.props.digit.is3}</td>
        <td style={{color:color4}}>{"4:"+this.props.digit.is4}</td>
        <td style={{color:color5}}>{"5:"+this.props.digit.is5}</td>
        <td style={{color:color6}}>{"6:"+this.props.digit.is6}</td>
        <td style={{color:color7}}>{"7:"+this.props.digit.is7}</td>
        <td style={{color:color8}}>{"8:"+this.props.digit.is8}</td>
        <td style={{color:color9}}>{"9:"+this.props.digit.is9}</td>
        <td>{this.props.digit.std}</td>
      </tr>
    );
  }
});

var DigitBoard = React.createClass({
  getInitialState: function(){
    return {digits: []};
  },
  componentWillMount: function(){
    var that = this;
    $.ajax({
      url: "http://69.164.193.25/digitapi",
      dataType: 'json',
      success: function(response){
        console.log(response);
        that.setState({digits: response});
      }
    })
  },
  render: function(){

    var digitBlocks = [];
    this.state.digits.forEach(function(digit){
      digitBlocks.push(<DigitBlock digit={digit} />);
    });

    return (
      <table>
        <tr>
          <th>Image</th>
          <th>Whiten Image</th>
          <th>0</th>
          <th>1</th>
          <th>2</th>
          <th>3</th>
          <th>4</th>
          <th>5</th>
          <th>6</th>
          <th>7</th>
          <th>8</th>
          <th>9</th>
          <th>std</th>
        </tr>
      {digitBlocks}
      </table>
    );
  }
});

React.render(
  <DigitBoard />,
  document.getElementById("app")    
)
