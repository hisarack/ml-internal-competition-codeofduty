var express = require('express');
var router = express.Router();
var Sequelize = require('sequelize');
var u = require('underscore');

var sequelize = new Sequelize('codeofduty', 'root', 'billychu21');
var Digit = sequelize.define('digit', {
  fname: {
    type: Sequelize.STRING,
    field: 'fname',
    primaryKey: true
  },
  is0: Sequelize.FLOAT,
  is1: Sequelize.FLOAT,
  is2: Sequelize.FLOAT,
  is3: Sequelize.FLOAT,
  is4: Sequelize.FLOAT,
  is5: Sequelize.FLOAT,
  is6: Sequelize.FLOAT,
  is7: Sequelize.FLOAT,
  is8: Sequelize.FLOAT,
  is9: Sequelize.FLOAT,
  std: Sequelize.FLOAT
},{
  tableName: 'digit',
  timestamps: false
});

/* GET home page. */
router.get('/digit', function(req, res, next) {

  Digit.findAll({}).then(function(digitsObj){
    var digits = u.chain(digitsObj)
     .map(function(digitObj){
      return digitObj.dataValues;
     });
    res.json(digits);
  });
});

module.exports = router;
