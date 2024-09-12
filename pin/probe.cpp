#include "pin.H"
#include <iostream>
using std::cerr;
using std::cout;
using std::dec;
using std::endl;
using std::flush;
using std::hex;
using std::string;
 
typedef int (*blank_op)();

std::string FuncName;

KNOB<std::string> KnobOpName(KNOB_MODE_WRITEONCE, "pintool",
 "FuncName", "default", "Specify the operation name");
 
int str_incl(const CHAR* str1, const CHAR* str2)
{
    if(strstr(str2, str1) == NULL)
        return 1;
    else
        return 0;
}

// This is the replacement routine.
//
int BlankOp(blank_op orgFuncptr, ADDRINT returnIp)
{
    // Normally one would do something more interesting with this data.
    //
    cout << "======instrumented=======" << endl << flush;
 
    // Call the relocated entry point of the original (replaced) routine.
    //
    // VOID* v = orgFuncptr(arg0);
    int v = 0;
 
    return v;
}
 
// Pin calls this function every time a new img is loaded.
// It is best to do probe replacement when the image is loaded,
// because only one thread knows about the image at this time.
//
VOID ImageLoad(IMG img, VOID* v)
{
    RTN rtn = RTN_FindByName(img, FuncName.c_str());

    if (RTN_Valid(rtn))
    {
        cout << "found the function => " << FuncName.c_str() << endl;
        if (RTN_IsSafeForProbedReplacement(rtn))
        {
            // cout << "Replacing malloc in " << IMG_Name(img) << endl;

            // Define a function prototype that describes the application routine
            // that will be replaced.
            //
            PROTO proto_malloc = PROTO_Allocate(PIN_PARG(int), CALLINGSTD_DEFAULT, FuncName.c_str(), PIN_PARG_END());

            // Replace the application routine with the replacement function.
            // Additional arguments have been added to the replacement routine.
            //
            RTN_ReplaceSignatureProbed(rtn, AFUNPTR(BlankOp), IARG_PROTOTYPE, proto_malloc, IARG_ORIG_FUNCPTR,
                                    IARG_FUNCARG_ENTRYPOINT_VALUE, 0, IARG_RETURN_IP, IARG_END);

            // Free the function prototype.
            //
            PROTO_Free(proto_malloc);
        }
        else
        {
            cout << "Skip replacing the func in " << IMG_Name(img) << " since it is not safe." << endl;
        }
    }
}
 
/* ===================================================================== */
/* Print Help Message                                                    */
/* ===================================================================== */
 
INT32 Usage()
{
    cerr << "This tool demonstrates how to replace an original" << endl;
    cerr << " function with a custom function defined in the tool " << endl;
    cerr << " using probes.  The replacement function has a different " << endl;
    cerr << " signature from that of the original replaced function." << endl;
    cerr << endl << KNOB_BASE::StringKnobSummary() << endl;
    return -1;
}
 
/* ===================================================================== */
/* Main: Initialize and start Pin in Probe mode.                         */
/* ===================================================================== */
 
int main(INT32 argc, CHAR* argv[])
{
    // Initialize symbol processing
    //
    PIN_InitSymbols();
 
    // Initialize pin
    //
    if (PIN_Init(argc, argv)) return Usage();

    FuncName = KnobOpName.Value();
    std::cout << "FuncName: " << FuncName << std::endl;
 
    // Register ImageLoad to be called when an image is loaded
    //
    IMG_AddInstrumentFunction(ImageLoad, 0);
 
    // Start the program in probe mode, never returns
    //
    PIN_StartProgramProbed();
 
    return 0;
}