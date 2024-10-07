#!/usr/bin/env python3
""" The skeleton for writing an interpreter given the bytecode.
"""

from dataclasses import dataclass
from pathlib import Path
import sys, logging
from typing  import Literal, TypeAlias, Optional, Union

l = logging
l.basicConfig(level=logging.DEBUG, format="%(message)s")

JvmType: TypeAlias = Union[Literal["boolean"], Literal["int"]]


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

    def create_interpreter(self, inputs):
        method = self.load()
        return SimpleInterpreter(
            bytecode=method["code"]["bytecode"],
            locals=inputs,
            stack=[],
            method_stack = [{}],
            heap = {},
            pc=0,
        )


@dataclass
class SimpleInterpreter:
    bytecode: list
    locals: list
    stack: list
    heap: dict
    pc: int
    method_stack: list[dict]
    done: Optional[str] = None

    def interpet(self, limit=100):
        for i in range(limit):
            next = self.bytecode[self.pc]
            l.debug(f"STEP {i}:")
            l.debug(f"  PC: {self.pc} {next}")
            l.debug(f"  LOCALS: {self.locals}")
            l.debug(f"  STACK: {self.stack}")
            l.debug(f"  HEAP: {self.heap}")
            l.debug(f" METHODS: {self.method_stack}")
            if fn := getattr(self, "step_" + next["opr"], None):
                fn(next)
            else:
                return f"can't handle {next['opr']!r}"

            if self.done:
                break

        else:
            self.done = "out of time"

        l.debug(f"DONE {self.done}")
        l.debug(f"  LOCALS: {self.locals}")
        l.debug(f"  STACK: {self.stack}")
        l.debug(f"  HEAP: {self.heap}")
        l.debug(f" METHODS: {self.method_stack}")
        return self.done

    def step_push(self, bc):
        try: self.stack.insert(0, bc["value"]["value"])
        except TypeError as e: 
            l.debug(f"Adding null element to stack")
            self.stack.insert(0, None)
        self.pc += 1

    def step_return(self, bc):
        if bc["type"] is not None:
            self.stack.pop(0)
        self.done = "ok"

    def step_binary(self, bc):

        if len(self.stack) < 2:
            raise ValueError("Not enough values on the stack for binary operation")

        b = self.stack.pop(0)
        a = self.stack.pop(0)

        opr = bc["operant"]
        if opr == "add":
            result = a + b
        elif opr == "sub":
            result = a - b
        elif opr == "mul":
            result = a * b
        elif opr == "div": 
            if b == 0:
                l.debug("Exception thrown: Division by zero")
                self.done = "divide by zero" # Stop execution
                return
            else:
                result = int(a / b)
        else:
            raise ValueError(f"Unknown binary operator: {opr}")

        self.stack.insert(0, result)
        self.pc += 1

    def step_load(self,bc):
        if bc["type"] == "const":
            constant = bc["value"]
            self.stack.append(constant)

        elif bc["type"] == "var":
            var_index = bc["index"]
            if var_index < len(self.locals):
                self.stack.insert(0,self.locals[var_index])
            else:
                raise ValueError(f"Variable index {var_index} out of range.")
            
        elif bc["type"] == "int":
            int_index = bc["index"]
            if int_index < len(self.locals):
                self.stack.insert(0,self.locals[int_index])
            else:
                raise ValueError(f"Integer index {int_index} out of range.")

        elif bc["type"] == "ref": # TODO 1
            ref_index = bc["index"]
            if ref_index < len(self.locals):
                self.stack.insert(0,self.locals[ref_index])
            else:
                raise ValueError(f"Reference index {ref_index} out of range.")

        else:
            raise ValueError(f"Unknown load type: {bc['type']}")

        self.pc += 1

    def step_ifz(self,bc):
        #Pop first value on stack
        value = self.stack.pop(0)
        condition = bc["condition"]
        #If value is zero, move to target index, if not zero, move to next instruction
        #Not sure what ne is here, maybe negative. More checking should be done for this function

        # I did some messing around here. 'ne' is 'not equal' and this can also be e.g. 'gt'
        # but this returns correctly for the assertion functions, 
        # we might have to add more options later - Kristine
        if value == None:
            l.debug(f"Exception thrown: Attempting to compare null value")
            self.done = "null pointer"
        elif condition == 'ne':
            if value != 0:
                self.pc = bc["target"]
            else:
                self.pc += 1
        elif condition == 'gt':
            if value > 0:
                self.pc = bc["target"]
            else:
                self.pc += 1
        elif condition == "eq":
            if value == 0:
                self.pc = bc["target"]
            else:
                self.pc += 1
        else:
            raise NotImplementedError(f"Unknown condition: {condition}")
        
    def step_if(self,bc):
        valueB = self.stack.pop(0)
        valueA = self.stack.pop(0)
        condition = bc["condition"]

        if condition == 'gt':
            if valueA > valueB:
                self.pc = bc["target"]
            else:
                self.pc += 1
        elif condition == 'eq':
            if valueA == valueB:
                self.pc = bc["target"]
            else:
                self.pc += 1
        elif condition == 'ge':
            if valueA >= valueB:
                self.pc = bc["target"]
            else:
                self.pc += 1
        elif condition == 'ne':
            if valueA != valueB:
                self.pc = bc["target"]
            else:
                self.pc += 1
        else:
            raise NotImplementedError(f"Unknown condition: {condition}")

    def step_get(self,bc):
        is_static = bc["static"]
        field = bc["field"] # {'class': 'jpamb/cases/Arrays', 'name': '$assertionsDisabled', 'type': 'boolean'}
        ## TODO XXX HERE
        self.stack.insert(0, value)
        self.pc += 1

    def step_new(self, bc):
        class_name = bc["class"]

        new_object = {
            "class": class_name,
            "initialized": False
        }

        obj_id = len(self.heap)
        self.heap[obj_id] = new_object

        self.stack.insert(0, obj_id)
        self.pc += 1

    def step_dup(self, bc):

        if not self.stack:
            raise ValueError("Stack is empty; cannot duplicate value.")

        top_value = self.stack[0]
        self.stack.insert(0, top_value)

        self.pc += 1

    def step_invoke(self, bc):
        method = bc["method"]
        method_name = method["name"]

        if method_name == "<init>":
            obj_id = self.stack.pop(0)
            # Initialize object in heap
            self.heap[obj_id]["initialized"] = True
        else:
            raise NotImplementedError(f"Method {method_name} not implemented.")

        self.pc += 1

    def step_throw(self, bc):
        ref = self.heap[self.stack.pop(0)]
        exception = ref['class']
        l.debug(f"Exception thrown: {exception}")
        if ref['initialized'] == True:
            match exception:
                case "java/lang/AssertionError":
                    self.done = "assertion error" # Stop execution
        self.pc += 1

    def step_newarray(self, bc):
        type = bc["type"]
        dim = bc["dim"]
        length = self.stack.pop()

        new_array = {
            "type": type,
            "dim": dim,
            "length": length,
            "elements": [0] * length
        }

        array_ref = len(self.heap)
        self.heap[array_ref] = new_array

        self.stack.insert(0, array_ref)
        self.pc += 1

    def step_array_store(self, bc):
        type = bc["type"]
        value = self.stack.pop(0)
        index = self.stack.pop(0)
        array_ref = self.stack.pop(0)

        if array_ref == None:
            l.debug(f"Exception thrown: Attempting to store to non initilized array")
            self.done = "null pointer"  # Stop execution

        else:
            n = self.heap[array_ref]["length"]

            if type != self.heap[array_ref]["type"]:
                raise ValueError("Type mismatch in array store operation.")

            if index < 0 or index >= n:
                l.debug(f"Exception thrown: Array out of bounds")
                self.done = "out of bounds" # Stop execution

            else: 
                l.debug(f"arr_ref: {array_ref}, index: {index}, value: {value}")
                self.heap[array_ref]["elements"][index] = value
        self.pc += 1

    def step_store(self, bc):
        index = bc["index"] # int
        type = bc["type"] # ref   

        if type == "ref" or type == "int":
            self.locals.insert(index, self.stack.pop(0))
        else:
            raise NotImplementedError(f"Unknown store type: {type}")
        
        self.pc += 1

    def step_arraylength(self, bc):
        array_ref = self.stack.pop(0)
        if array_ref != None:
            length = self.heap[array_ref]["length"]
            self.stack.insert(0, length)
        else:
            l.debug(f"Exception thrown: Attempting to get length of non initialized array")
            self.done = "null pointer" # Stop execution
        self.pc += 1

    def step_array_load(self, bc):
        type = bc["type"]

        arr_ref = self.stack.pop()
        index = self.stack.pop()

        l.debug(f"arr_ref: {arr_ref}, index: {index}")

        val = self.heap[arr_ref]["elements"][index]

        self.stack.append(val)

        self.pc += 1

    def step_goto(self, bc):
        self.pc = bc["target"]

if __name__ == "__main__":
    methodid = MethodId.parse(sys.argv[1])
    inputs = []
    result = sys.argv[2][1:-1]
    if result != "":
        for i in result.split(","):
            if i == "true" or i == "false":
                inputs.append(i == "true")
            else:
                inputs.append(int(i))
    print(methodid.create_interpreter(inputs).interpet())
