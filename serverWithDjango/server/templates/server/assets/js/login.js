$('.message a').click(function(){
   $('form').animate({height: "toggle", opacity: "toggle"}, "slow");
});

function type(d, columns) {
	d.pass = parseInt(d.pass);
	for (var i = 1, n = columns.length, c; i < n; ++i) d[c = columns[i]] = +d[c];
	return d;
}

d3.csv("assets/pass.csv", type, function(error, data){
	console.log(data);
	var d = data;
});
var lk = document.getElementById("link");
function pass(ps) {
	var ps = ps.value;
	if (ps == 22) {
		lk.href = "first.html";
	}
}