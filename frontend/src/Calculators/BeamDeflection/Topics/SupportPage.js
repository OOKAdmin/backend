import React from 'react'

export default function SupportPage(props) {
  return (
    <>
      <h2 className='claculator-conversation-title' style={{ textAlign: 'center', fontWeight: '600', fontSize: '1.4vw' }}>Support</h2>
      <br />
      {props.Support}
      {props.addSupportBtn}
      <br />
      {props.SumbitBtn}
    </>
  )
}
