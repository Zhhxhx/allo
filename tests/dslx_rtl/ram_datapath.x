// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

pub proc ram_check {
    command: chan<u20> in;
    answer: chan<u16> out;
    first_req: chan<u19> out;
    first_resp: chan<u16> in;
    second_req: chan<u19> out;
    second_resp: chan<u16> in;

    config(command: chan<u20> in, answer: chan<u16> out,
           first_req: chan<u19> out, first_resp: chan<u16> in,
           second_req: chan<u19> out, second_resp: chan<u16> in) {
        (command, answer, first_req, first_resp, second_req, second_resp)
    }

    init { () }

    next(state: ()) {
        let (tok, cmd) = recv(join(), command);
        let bank = cmd[19+:u1] == u1:1;
        let payload = cmd[0+:u19];
        let read = payload[18+:u1] == u1:0;
        let tok = send_if(tok, first_req, !bank, payload);
        let tok = send_if(tok, second_req, bank, payload);
        let (tok, a) = recv_if(tok, first_resp, read && !bank, u16:0);
        let (tok, b) = recv_if(tok, second_resp, read && bank, u16:0);
        let tok = send_if(tok, answer, read, if bank { b } else { a });
    }
}
