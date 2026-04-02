import React from 'react'

export default function LoadPage(props) {
  return (
    <>
        <br/>
        {props.DistributedLoad}
        {props.addDistributedLoad}

        {props.PointLoad}
        <br/>
        {props.AddPointLoad}
        {props.SumbitBtn}
    </>
  )
}
