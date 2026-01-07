var userName: string = "John";
var userAge2: number = 20;
var userHeight2: number = 180;

const addTwoNum = (num1: number, num2: number): number => {
    var result = num1 + num2;
    return result;
}

console.log(addTwoNum(1, 2));

// 객체에 대한 기준을 만드는 것 : 인터페이스
interface Member {
    name: string;
    age: number;
    pw: string;
}

var member: Member = {
    name: '선우',
    age: 20,
    pw: '1234'
}