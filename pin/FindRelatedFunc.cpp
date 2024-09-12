#include "pin.H"
#include <iostream>
#include <fstream>
using std::cerr;
using std::cout;
using std::dec;
using std::endl;
using std::flush;
using std::hex;
using std::string;

/* Global Variables */
std::ofstream TraceFile;

int func_count = 0;
std::vector<std::string> related_func;
std::string OpName;
 
int str_incl(const CHAR* str1, const CHAR* str2)
{
    if(strstr(str2, str1) == NULL)
        return 1;
    else
        return 0;
}

bool endsWith(const std::string& mainStr, const std::string& toMatch) {
    if (mainStr.size() >= toMatch.size() &&
        mainStr.compare(mainStr.size() - toMatch.size(), toMatch.size(), toMatch) == 0) {
        return true;
    } else {
        return false;
    }
}

/* Commandline Switches */
// KNOB<string> KnobOutputFile(KNOB_MODE_WRITEONCE, "pintool",
//     "opname", "default", "o", "Used_Func.txt", "specify trace file name");

KNOB<std::string> KnobOpName(KNOB_MODE_WRITEONCE, "pintool", "opname", "default", "Specify the operation name");

VOID beforeCall(CHAR* name)
{
    func_count++;
    related_func.push_back(name);
    TraceFile << name << endl;
}

/* Print return value of the function. */
VOID afterCall(ADDRINT ret)
{
    // TraceFile << "  return: " << ret << endl;
}

/* Instrument the functions */
VOID InstFunc(IMG& img, SYM& sym, const char* funcName, AFUNPTR bfFunPtr, AFUNPTR afFunPtr)
{
    RTN funcRtn = RTN_FindByAddress(IMG_LowAddress(img) + SYM_Value(sym));
    if (RTN_Valid(funcRtn))
    {
        RTN_Open(funcRtn);
        // get function's argument
        RTN_InsertCall(
            funcRtn,
            IPOINT_BEFORE, bfFunPtr,
            IARG_ADDRINT, funcName,
            IARG_END);
        RTN_Close(funcRtn);
    }
}

/* Instrumentation routines */
VOID Image(IMG img, VOID* V )
{
    // Walk through the symbols in the symbol table.
    for (SYM sym = IMG_RegsymHead(img); SYM_Valid(sym); sym = SYM_Next(sym))
    {
        string undFuncName = PIN_UndecorateSymbolName(SYM_Name(sym), UNDECORATION_NAME_ONLY);

        std::string ori_opname = OpName;
        std::string exclude_func=OpName;
        std::string exc_name = "<";
        exclude_func.append(exc_name);
        if (!str_incl(OpName.c_str(), undFuncName.c_str()))
        {
            if (endsWith(undFuncName, "Eval") || !str_incl("Eval<", undFuncName.c_str()))
            {
                const char* funcName;
                funcName = strdup(undFuncName.c_str());
                // cout << "related_name: " << funcName << endl;
                InstFunc(img, sym, funcName, (AFUNPTR)beforeCall, (AFUNPTR)afterCall);
            }
        }
    }
}

VOID Fini(INT32 code, VOID* v)
{
    // TraceFile.close();
    // for (int i = 0; i < func_count; i++)
    // {
    //     // cout << "num of func: " << func_count << endl;
    //     cout << "related_func: " << related_func[i] << endl;
    // }
    TraceFile.close();
}

 
/* ===================================================================== */
/* Print Help Message                                                    */
/* ===================================================================== */
 
INT32 Usage()
{
    cerr << "This tool demonstrates how to find executed functions" << endl;
    cerr << " for a specific operator " << endl;
    cerr << endl << KNOB_BASE::StringKnobSummary() << endl;
    return -1;
}
 
/* ===================================================================== */
/* Main: Initialize and start Pin in Probe mode.                         */
/* ===================================================================== */
 
int main(INT32 argc, CHAR* argv[])
{
    // Initialize pin & symbol manager
    PIN_InitSymbols();
    if (PIN_Init(argc, argv))
    {
        return Usage();
    }

    OpName = KnobOpName.Value();
    std::cout << "OpName: " << OpName << std::endl;

    // Write to a file since cout and cerr maybe closed by the application
    TraceFile.open("Used_Func.txt");
    TraceFile << std::hex << std::showbase;


    // Register Image to be called to instrument functions.
    IMG_AddInstrumentFunction(Image, 0);
    PIN_AddFiniFunction(Fini, 0);

    // Never returns
    PIN_StartProgram();
 
    return 0;
}
