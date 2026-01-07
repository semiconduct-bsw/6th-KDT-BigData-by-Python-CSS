//console.log("Hello world");

// 데이터
// var = 재선언 가능, 재할당 가능
var userName = "John";
userName = "John";  // 할당
var userName = "Jane";  // 선언

// let = 재선언 불가능, 재할당 가능
let userAge = 20;
userAge = 21;  // 할당은 가능

// const = 재선언 불가능, 재할당 불가능
const userHeight = 180;

// 여러개 받는 타입 : 배열(=python의 리스트), 객체(오브젝트)
var userArray = ["John", "Jane", "Jim"];
var one = userArray[0];

var user = {
    name: 'John',
    age: 20,
    height: 180
}
var height = user['height'];
var height = user.height;

// 함수
function addTwoNumbers(num1, num2) {
    var c = num1 + num2;
    console.log(c);

    return c;
}

var result = addTwoNumbers(10, 20);
console.log(result);

// arrow function , 화살표 함수
var addTwoNumbersArrow = (numA, numB) => {
    var d = numA + numB;
    console.log(d);

    return d;
}

var resultArrow = addTwoNumbersArrow(10, 20);
console.log(resultArrow);

// 조건문
function checkScore(score) {
    if (score >= 90) {
        console.log("A");
    } else if (score >= 80) {
        console.log("B");
    } else {
        console.log("C");
    }
}
checkScore(80);

// 반복문
// for (var i = 0; i < 10; i++) {
//     console.log(i);
// }

// while (i < 10) {
//     console.log(i);
//     i++;
// }

var users = ["John", "Jane", "Jim"];
for (var user of users) {
    console.log(user);
}

// 클래스
class Animal {
    constructor(name, age) {
        this.name = name;
        this.age = age;
    }

    sayHello() {
        console.log("Hello " + this.name);
    }
}

var dog = new Animal("dog", 10);
console.log(dog.name);
console.log(dog.age);
dog.sayHello();