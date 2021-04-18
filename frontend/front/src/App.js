import React, {useEffect, useState} from 'react'
import styled from "styled-components";
import './index.css';
import Graph from "./components/graph";
import {data} from "./data";

const pluser = 10;

function App() {
    const [index, setIndex] = useState(0);
    const sample_data = data.slice(parseInt(index), parseInt(index) + pluser);
    useEffect(() => {
        let myInterval = setInterval(() => index + 1 + pluser >= data.length ? setIndex(0) : setIndex(index + 1), 166,6666666666667);
        return () => {
            clearInterval(myInterval);
        };
    });

    const objects_count = sample_data.map((elem) => elem['objects_count']);
    const areas_mean = sample_data.map((elem) => elem['areas_mean']);
    const areas_var = sample_data.map((elem) => elem['areas_var']);
    const mean_distance = sample_data.map((elem) => elem['mean_distance']);
    const var_distance = sample_data.map((elem) => elem['var_distance']);
    const sum_vec_length = sample_data.map((elem) => elem['sum_vec_length']);
    const sum_vec_angle = sample_data.map((elem) => elem['sum_vec_angle']);
    const warns = sample_data.map((elem) => elem['warnings']);

    const warnings = []
    for (let elems in warns)
        for (let elem in elems)
            if (!warnings.includes(elem))
                warnings.push(elem);
    return (
        <AppWrapper>
            <Img src={process.env.PUBLIC_URL + '/imgs/' + index * 5 + '.png'} />
            <Flexer>
                <Graph data={{
                    name: "objects_count",
                    description: "",
                    vals: objects_count,
                    warn: warnings.includes('objects_count')
                }}/>
                <Graph data={{
                    name: "areas_mean",
                    description: "",
                    vals: areas_mean,
                    warn: warnings.includes('areas_mean')
                }}/>
                <Graph data={{
                    name: "areas_var",
                    description: "",
                    vals: areas_var,
                    warn: warnings.includes('areas_var'),
                    isyellow: true,
                    isred: true
                }}/>
                <Graph data={{
                    name: "mean_distance",
                    description: "",
                    vals: mean_distance,
                    warn: warnings.includes('mean_distance'),
                    isyellow: true,
                    isred: false
                }}/>
                <Graph data={{
                    name: "var_distance",
                    description: "",
                    vals: var_distance,
                    warn: warnings.includes('var_distance'),
                    isyellow: false,
                    isred: true
                }}/>
                <Graph data={{
                    name: "sum_vec_length",
                    description: "",
                    vals: sum_vec_length,
                    warn: warnings.includes('sum_vec_length')
                }}/>
                <Graph data={{
                    name: "sum_vec_angle",
                    description: "",
                    vals: sum_vec_angle,
                    warn: warnings.includes('sum_vec_angle')
                }}/>
            </Flexer>
        </AppWrapper>
    );
}

const AppWrapper = styled.div`
  max-width: 1500px;
  margin: 0 auto;
`;
const Flexer = styled.div`
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
`;
const Img = styled.img`
  margin: 0 auto;
  display: block;
  width: 600px;
  height: 450px;
  margin-bottom: 20px;
`;

export default App;
