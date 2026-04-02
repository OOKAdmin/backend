import React, { useState } from 'react'

export default function InputParametersOfShackle() {
const [PadeyeInputParametersDropDown, setPadeyeInputParametersDropDown] = useState(false);

    const TogglePadeyeInputParametersDropDown = () => {
        setPadeyeInputParametersDropDown(!PadeyeInputParametersDropDown);
    };
    return (
        <>
            <h3 className='calculator-info-blue-section-main-topic MainHeadingOfDropdown' style={{
                padding: '0',
                margin: 'auto',
                width: '90%',
                justifyContent: 'left',
                textTransform: 'uppercase',
                fontWeight: '300',
                fontSize: '3.65vw',
                important: true,
            }}>
                Input parameters :
            </h3>
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'flex-start', width: '95%', margin: 'auto' }}>

                <div>
                    <h3 className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '90%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Shackle Safe Working Load (Swl) : </h3>
                    <h3 className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '90%', justifyContent: 'left' }}>Maximum load that shackle can handle safely.</h3>
                    <br />
                    <h3 className='calculator-info-blue-section-main-topic TopicOfDropdown' style={{ padding: '0', margin: 'auto', width: '90%', justifyContent: 'left', fontSize: '2.2vw', important: true }}>Shackle pin diameter :</h3>
                    <h3 className='calculator-info-blue-section-main-topic DefinatinOfDropDown' style={{ padding: '0', margin: 'auto', width: '90%', justifyContent: 'left' }}>Shackle pin diameter must match the Padeye hole diameter, with a clearance not exceeding 3% of the pin diameter.</h3>
                    <br />
                </div>
            </div>
        </>
  )
}
