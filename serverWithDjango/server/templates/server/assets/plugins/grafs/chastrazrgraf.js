var svg = d3.select("svg"),
    margin = {top: 20, right: 80, bottom: 30, left: 50},
    width = svg.attr("width") - margin.left - margin.right,
    height = svg.attr("height") - margin.top - margin.bottom,
    g = svg.append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");

function type(d, _, columns) {
  d.date = parseFloat(d.date);
  d.power = parseFloat(d.power);
  d.impulse = parseInt(d.impulse);
  for (var i = 1, n = columns.length, c; i < n; ++i) d[c = columns[i]] = +d[c];
  return d;
}

d3.csv("chr.csv", type, function(error, data) {
  if (error) throw error;
  
console.log(data);
var circles = svg.selectAll("circle")
    .data(data)
    .enter()
    .append("circle")
    .attr("fill",function(d){
      if (d.power <= 60) {
        return "green";
      } else if (d.power <= 160) {
        return "yellow";
      } else {return "red";}
    })
    .attr("stroke","black")
    .attr("stroke-width",1)
    .attr("cx",function(d){return d.date*40})
    .attr("cy",function(d){return 500-d.impulse/4.825;})
    .attr("r",function(d){return d.power/10;});


});