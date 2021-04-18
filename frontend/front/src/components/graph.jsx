import React, { useState, useEffect } from 'react'
import styled from 'styled-components'
import {Line} from "react-chartjs-2";

const indicies = ['', '', '', '', '', '', '', '', '', ''];

export default function Graph({data}) {
    const {name, description, vals, warn, isred, isyellow} = data;
    const [max, setMax] = useState(Math.max(...vals));
    useEffect(() => Math.max(...vals) > parseInt(max) && setMax(Math.floor(Math.max(...vals)) + 1), [max, vals]);
    return (
        <Wrapper>
            <Flexer>
                <div style={{width: '80%'}}>
                    <Name>{name}</Name>
                    <Description>{description}</Description>
                </div>
                <Colorizer color={isyellow ? 'yellow' : '#F0F0F0'}/>
                <Colorizer color={isred ? 'red' : '#F0F0F0'}/>
            </Flexer>
            <Line options={{
                legend: {display: false}, animation: {duration: 0}, scales: {
                    yAxes: [{
                        display: true,
                        ticks: {
                            beginAtZero: true,
                            max
                        }
                    }]
                }
            }} data={{
                labels: indicies,
                datasets: [{
                    data: vals,
                    lineTension: 0.01
                }]
            }
            }/>
        </Wrapper>
    );
}

const Wrapper = styled.div`
  background-color: white;
  border-radius: 5px;
  padding: 10px;
  margin: 10px;
  width: 25%;
`;
const Name = styled.div`
  font-size: 20px;
  margin-bottom: 10px;
`;
const Description = styled.div`
  margin-bottom: 10px;
`;
const Flexer = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
`;
const Colorizer = styled.div`
  margin-left: 10px;
  width: 30px;
  height: 30px;
  border-radius: 5px;
  background-color: ${({color}) => color}};
`;
