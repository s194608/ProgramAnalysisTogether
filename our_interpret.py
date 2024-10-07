#!/usr/bin/env python
""" The skeleton for writing an interpreter given the bytecode.
"""

import random
from dataclasses import dataclass
from pathlib import Path
import sys, logging
from typing  import Literal, TypeAlias, Optional, Union

l = logging
l.basicConfig(level=logging.DEBUG, format="%(message)s")

JvmType: TypeAlias = Union[Literal["boolean"], Literal["int"]]

def generate_random_inputs(params):
    inputs = []
    for param in params:
        if param == "boolean":
            inputs.append(random.choice([True, False]))
        elif param == "int":
            inputs.append(random.randint(0, 100))  # Arbitrary range for integers
        else:
            raise ValueError(f"Unknown parameter type: {param}")
    return inputs


@dataclass(frozen=True)
class MethodId:
    class_name: str
    method_name: str
    params: list[JvmType]
    return_type: Optional[JvmType]

    @classmethod
    def parse(cls, name):
        import re

        TYPE_LOOKUP: dict[str, JvmType] = {
            "Z": "boolean",
            "I": "int",
        }

        RE = (
            r"(?P<class_name>.+)\.(?P<method_name>.*)\:\((?P<params>.*)\)(?P<return>.*)"
        )
        if not (i := re.match(RE, name)):
            l.error("invalid method name: %r", name)
            sys.exit(-1)

        return cls(
            class_name=i["class_name"],
            method_name=i["method_name"],
            params=[TYPE_LOOKUP[p] for p in i["params"]],
            return_type=None if i["return"] == "V" else TYPE_LOOKUP[i["return"]],
        )

    def classfile(self):
        print(self)
        return Path('.\decompiled\jpamb\cases\Arrays.json')

    def load(self):
        import json

        classfile = self.classfile()
        with open(classfile) as f:
            l.debug(f"read decompiled classfile {classfile}")
            classfile = json.load(f)
        for m in classfile["methods"]:
            if (
                m["name"] == self.method_name
                and len(self.params) == len(m["params"])
                and all(
                    p == t["type"]["base"] for p, t in zip(self.params, m["params"])
                )
            ):
                return m
        else:
            print("Could not find method")
            sys.exit(-1)

    def create_interpreter(self, inputs, arithmetic):
        method = self.load()
        state = InterpreterState(
            bytecode=method["code"]["bytecode"],
            locals=inputs,
            stack=[],
            method_stack=[{}],
            heap={},
            pc=0
        )
        return Interpreter(state=state, arithmetic=arithmetic)


@dataclass
class InterpreterState:
    bytecode: list
    locals: list
    stack: list
    heap: dict
    pc: int
    method_stack: list[dict]
    done: Optional[str] = None


@dataclass
class Arithmetic:
    def add(self, state: InterpreterState, type_, a, b):
        if type_ == "int":
            return a + b
        raise ValueError(f"Unsupported type for addition: {type_}")

    def sub(self, state: InterpreterState, type_, a, b):
        if type_ == "int":
            return a - b
        raise ValueError(f"Unsupported type for subtraction: {type_}")

    def mul(self, state: InterpreterState, type_, a, b):
        if type_ == "int":
            return a * b
        raise ValueError(f"Unsupported type for multiplication: {type_}")

    def div(self, state: InterpreterState, type_, a, b):
        if type_ == "int":
            if b == 0:
                l.debug("Exception thrown: Division by zero")
                state.done = "divide by zero"  # Stop execution
                return
            else:
                return a // b
        raise ValueError(f"Unsupported type for division: {type_}")

@dataclass
@dataclass
class Interpreter:
    state: InterpreterState
    arithmetic: Arithmetic



    def interpret(self, limit=20):
        for i in range(limit):
            next = self.state.bytecode[self.state.pc]
            l.debug(f"STEP {i}:")
            l.debug(f"  PC: {self.state.pc} {next}")
            l.debug(f"  LOCALS: {self.state.locals}")
            l.debug(f"  STACK: {self.state.stack}")
            l.debug(f"  HEAP: {self.state.heap}")
            l.debug(f" METHODS: {self.state.method_stack}")
            if fn := getattr(self, "step_" + next["opr"], None):
                fn(next)
            else:
                return f"can't handle {next['opr']!r}"

            if self.state.done is not None:
                break

        else:
            self.state.done = "out of time"

        l.debug(f"DONE")
        l.debug(f"  LOCALS: {self.state.locals}")
        l.debug(f"  STACK: {self.state.stack}")
        l.debug(f"  HEAP: {self.state.heap}")
        l.debug(f" Done: {self.state.done}")
        return self.state.done

    def step_push(self, bc):
        try: self.state.stack.insert(0, bc["value"]["value"])
        except TypeError as e:
            l.debug(f"Adding null element to stack")
            self.state.stack.insert(0, None)
        self.state.pc += 1

    def step_return(self, bc):
        if bc["type"] is not None:
            self.state.stack.pop(0)
        self.state.done = "ok"

    def step_binary(self, bc):
        st = self.state
        a = st.stack.pop()
        b = st.stack.pop()
        opr = bc["operant"]

        if opr == "add":
            result = self.arithmetic.add(st, bc["type"], a, b)  # Pass state
        elif opr == "sub":
            result = self.arithmetic.sub(st, bc["type"], a, b)  # Pass state
        elif opr == "mul":
            result = self.arithmetic.mul(st, bc["type"], a, b)  # Pass state
        elif opr == "div":
            result = self.arithmetic.div(st, bc["type"], a, b)  # Pass state
        else:
            raise ValueError(f"Unknown binary operator: {opr}")

        if st.done:  # Check if done was set
            return  # Stop execution if done is set

        st.stack.append(result)
        st.pc += 1

    def step_push(self, bc):
        """Pushes a constant value onto the stack."""
        self.state.stack.insert(0, bc["value"]["value"])
        self.state.pc += 1

    def step_return(self, bc):
        """Handles return from a method."""
        if bc["type"] is not None:
            self.state.stack.pop(0)
        self.state.pc += 1  # Make sure to advance pc, or reset as needed
        self.state.done = "ok"

    def step_load(self,bc):
        if bc["type"] == "const":
            constant = bc["value"]
            self.state.stack.append(constant)

        elif bc["type"] == "var" or bc["type"] == "int" or bc["type"] == "ref":
            var_index = bc["index"]
            if var_index < len(self.state.locals):
                self.state.stack.insert(0,self.state.locals[var_index])
            else:
                raise ValueError(f"Variable index {var_index} out of range.")
            
        elif bc["type"] == "int":
            int_index = bc["index"]
            if int_index < len(self.state.locals):
                self.state.stack.insert(0,self.state.locals[int_index])
            else:
                raise ValueError(f"Integer index {int_index} out of range.")

        elif bc["type"] == "ref": # TODO 1
            ref_index = bc["index"]
            if ref_index < len(self.state.locals):
                self.state.stack.insert(0,self.state.locals[ref_index])
            else:
                raise ValueError(f"Reference index {ref_index} out of range.")

        else:
            raise ValueError(f"Unknown load type: {bc['type']}")

        self.state.pc += 1

    def step_ifz(self,bc):
        #Pop first value on stack
        value = self.state.stack.pop(0)
        condition = bc["condition"]
        #If value is zero, move to target index, if not zero, move to next instruction
        #Not sure what ne is here, maybe negative. More checking should be done for this function

        # I did some messing around here. 'ne' is 'not equal' and this can also be e.g. 'gt'
        # but this returns correctly for the assertion functions, 
        # we might have to add more options later - Kristine
        if value is None:
            l.debug(f"Exception thrown: Attempting to compare null value")
            self.state.done = "null pointer"
        elif condition == 'ne':
            if value != 0:
                self.state.pc = bc["target"]
            else:
                self.state.pc += 1
        elif condition == 'gt':
            if value > 0:
                self.state.pc = bc["target"]
            else:
                self.state.pc += 1
        elif condition == "eq":
            if value == 0:
                self.state.pc = bc["target"]
            else:
                self.state.pc += 1
        else:
            raise NotImplementedError(f"Unknown condition: {condition}")
        
    def step_if(self,bc):
        valueB = self.state.stack.pop(0)
        valueA = self.state.stack.pop(0)
        condition = bc["condition"]

        if condition == 'gt':
            if valueA > valueB:
                self.state.pc = bc["target"]
            else:
                self.state.pc += 1
        elif condition == 'eq':
            if valueA == valueB:
                self.state.pc = bc["target"]
            else:
                self.state.pc += 1
        elif condition == 'ge':
            if valueA >= valueB:
                self.state.pc = bc["target"]
            else:
                self.state.pc += 1
        elif condition == 'ne':
            if valueA != valueB:
                self.state.pc = bc["target"]
            else:
                self.state.pc += 1
        else:
            raise NotImplementedError(f"Unknown condition: {condition}")

    def step_get(self,bc):
        offset = bc["offset"]
        is_static = bc["static"]

        if is_static:
            # Retrieve from heap (static locals)
            if offset in self.state.heap:
                value = self.state.heap[offset]
            else:
                self.state.heap[offset] = None
                value = self.state.heap[offset]
        else:
            if offset < len(self.state.locals):
                value = self.state.locals[offset]
            else:
                raise IndexError(f"Offset {offset} out of range in dynamic locals")

        self.state.stack.insert(0, value)
        self.state.pc += 1

    def step_new(self, bc):
        class_name = bc["class"]

        new_object = {
            "class": class_name,
            "initialized": False
        }

        obj_id = len(self.state.heap)
        self.state.heap[obj_id] = new_object

        self.state.stack.insert(0, obj_id)
        self.state.pc += 1

    def step_dup(self, bc):

        if not self.state.stack:
            raise ValueError("Stack is empty; cannot duplicate value.")

        top_value = self.state.stack[0]
        self.state.stack.insert(0, top_value)

        self.state.pc += 1

    def step_invoke(self, bc):
        method = bc["method"]
        method_name = method["name"]

        if method_name == "<init>":
            obj_id = self.state.stack.pop(0)
            # Initialize object in heap
            self.state.heap[obj_id]["initialized"] = True
        else:
            raise NotImplementedError(f"Method {method_name} not implemented.")

        self.state.pc += 1

    def step_throw(self, bc):
        ref = self.state.heap[self.state.stack.pop(0)]
        exception = ref['class']
        l.debug(f"Exception thrown: {exception}")

        if ref['initialized'] == True:
            match exception:
                case "java/lang/AssertionError":
                    self.state.done = "assertion error" # Stop execution
        self.state.pc += 1

    def step_newarray(self, bc):
        type = bc["type"]
        dim = bc["dim"]
        length = self.state.stack.pop()

        new_array = {
            "type": type,
            "dim": dim,
            "length": length,
            "elements": [0] * length
        }

        array_ref = len(self.state.heap)
        self.state.heap[array_ref] = new_array

        self.state.stack.insert(0, array_ref)
        self.state.pc += 1

    def step_array_store(self, bc):
        type = bc["type"]
        value = self.state.stack.pop(0)
        index = self.state.stack.pop(0)
        array_ref = self.state.stack.pop(0)

        if array_ref == None:
            l.debug(f"Exception thrown: Attempting to store to non initilized array")
            self.state.done = "null pointer"  # Stop execution

        else:
            n = self.state.heap[array_ref]["length"]

            if type != self.state.heap[array_ref]["type"]:
                raise ValueError("Type mismatch in array store operation.")

            if index < 0 or index >= n:
                l.debug(f"Exception thrown: Array out of bounds")
                self.state.done = "out of bounds" # Stop execution

            else:
                l.debug(f"arr_ref: {array_ref}, index: {index}, value: {value}")
                self.state.heap[array_ref]["elements"][index] = value
        self.state.pc += 1

    def step_store(self, bc):
        index = bc["index"] # int
        type = bc["type"] # ref   

        if type == "ref" or type == "int":
            self.state.locals.insert(index, self.state.stack.pop(0))
        else:
            raise NotImplementedError(f"Unknown store type: {type}")

        self.state.pc += 1

    def step_arraylength(self, bc):
        array_ref = self.state.stack.pop(0)
        if array_ref != None:
            length = self.state.heap[array_ref]["length"]
            self.state.stack.insert(0, length)
        else:
            l.debug(f"Exception thrown: Attempting to get length of non initialized array")
            self.state.done = "null pointer" # Stop execution
        self.state.pc += 1

    def step_array_load(self, bc):
        type = bc["type"]

        arr_ref = self.state.stack.pop()
        index = self.state.stack.pop()

        l.debug(f"arr_ref: {arr_ref}, index: {index}")

        val = self.state.heap[arr_ref]["elements"][index]

        self.state.stack.append(val)

        self.state.pc += 1

    def step_goto(self, bc):
        self.state.pc = bc["target"]

def main_analysis(method_id, num_tests=100):
    arithmetic = Arithmetic()
    behaviors = run_interpreter_with_random_inputs(method_id, arithmetic, num_tests)

    behavior_counts = {
        "divide by zero": 0,
        "out of bounds": 0,
        "null pointer": 0,
        "ok": 0,
        "assertion error": 0,
    }

    for behavior in behaviors:
        behavior_counts[behavior] += 1

    # Emit results based on observed behaviors
    if behavior_counts["divide by zero"] > 0:
        print("divide by zero;100%")
    elif behavior_counts["out of bounds"] > 0:
        print("out of bounds;100%")
    elif behavior_counts["null pointer"] > 0:
        print("null pointer;100%")
    elif behavior_counts["assertion error"] > 0:
        print("assertion error;100%")
    else:
        print("Assertion error: 50%")  # If no specific behaviors, consider it normal

    # Check if it completed successfully without any errors
    if behavior_counts["ok"] == num_tests:
        print("ok")

def run_interpreter_with_random_inputs(method_id, arithmetic, num_tests):
    behaviors = []

    for _ in range(num_tests):
        inputs = generate_random_inputs(method_id.params)
        interpreter = method_id.create_interpreter(inputs, arithmetic)
        result = interpreter.interpret()

        # Collect the behaviors
        if result is not None:
            if result == "divide by zero":
                behaviors.append("divide by zero")
            elif result == "out of bounds":
                behaviors.append("out of bounds")
            elif result == "null pointer":
                behaviors.append("null pointer")
            elif result == "ok":
                behaviors.append("ok")
            else:
                behaviors.append("assertion error")
        else:
            behaviors.append("ok")  # If no result, consider it runs to completion

    return behaviors


if __name__ == "__main__":
    method_id = MethodId.parse(sys.argv[1])
    num_tests = 100  # Arbitrary number of tests to run
    main_analysis(method_id, num_tests)