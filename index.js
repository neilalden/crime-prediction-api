const input = {
	z: 35,
	x: 31,
	y: 1,
	u: 0,
	i: 8,
	w: 2,
};
fetch("http://neilalden.pythonanywhere.com/predict/crime", {
	// mode: "no-cors",
	method: "POST",
	headers: { "Content-Type": "application/json" },
	body: JSON.stringify(input),
})
	.then((response) => response.json())
	.then((data) => console.log(data));
